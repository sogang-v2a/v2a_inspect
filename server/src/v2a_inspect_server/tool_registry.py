from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from v2a_inspect.contracts import (
    CandidateCut,
    EvidenceWindow,
    LabelCandidate,
    MultitrackDescriptionBundle,
    TrackCrop,
)
from v2a_inspect.tools import (
    build_candidate_cuts,
    build_context_candidate_cuts,
    build_evidence_windows,
    evidence_windows_to_scene_boundaries,
    export_window_clips,
    generate_storyboard,
    group_entity_embeddings,
    hydrate_evidence_windows,
    probe_video,
    route_track,
    sample_frames,
)
from v2a_inspect.tools.types import CandidateGroup, EntityEmbedding, FrameBatch, Sam3EntityTrack, SceneBoundary

from .constants import DEFAULT_TRACK_LABELS, RECOVERY_SCENE_LABELS
from .crops import crop_tracks
from .settings import get_server_runtime_settings
from .reid import build_identity_edges, build_provisional_source_tracks
from .semantics import (
    build_ambience_beds,
    build_generation_groups,
    build_sound_event_segments,
)
from .descriptions import synthesize_canonical_descriptions
from .validators import validate_bundle


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: Callable[..., object]
    description: str


def _artifact_root(video_path: str) -> Path:
    settings = get_server_runtime_settings()
    preferred_root = (
        settings.shared_video_dir.parent / "artifacts"
        if settings.shared_video_dir is not None
        else Path(tempfile.gettempdir()) / "v2a_inspect_artifacts"
    )
    preferred_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f"{Path(video_path).stem}-{uuid4().hex[:8]}-",
            dir=str(preferred_root),
        )
    )


if TYPE_CHECKING:
    from .runtime import ToolingRuntime


def build_tool_registry(tooling_runtime: "ToolingRuntime") -> dict[str, ToolDefinition]:
    def structural_overview(
        *,
        video_path: str,
        target_scene_seconds: float = 5.0,
        frames_per_scene: int = 3,
        output_root: str | None = None,
    ) -> dict[str, object]:
        probe = probe_video(video_path)
        candidate_cuts = build_candidate_cuts(
            video_path,
            probe=probe,
            target_scene_seconds=target_scene_seconds,
        )
        evidence_windows = build_evidence_windows(
            probe=probe,
            candidate_cuts=candidate_cuts,
        )
        scenes = evidence_windows_to_scene_boundaries(
            evidence_windows,
            candidate_cuts=candidate_cuts,
        )
        frame_root = Path(output_root) if output_root is not None else _artifact_root(video_path)
        frame_root.mkdir(parents=True, exist_ok=True)
        frame_batches = sample_frames(
            video_path,
            scenes,
            output_dir=str(frame_root),
            frames_per_scene=frames_per_scene,
        )
        storyboard_path = generate_storyboard(
            frame_batches,
            output_path=str(frame_root / "storyboard.jpg"),
        )
        clip_paths_by_window = export_window_clips(
            video_path,
            evidence_windows,
            output_dir=str(frame_root / "clips"),
            max_windows=3,
        )
        evidence_windows = hydrate_evidence_windows(
            evidence_windows,
            frame_batches,
            storyboard_path=storyboard_path,
            clip_paths_by_window=clip_paths_by_window,
        )
        return {
            "probe": probe,
            "candidate_cuts": candidate_cuts,
            "evidence_windows": evidence_windows,
            "frame_batches": frame_batches,
            "storyboard_path": storyboard_path,
            "artifact_root": str(frame_root),
            "frames_per_scene": frames_per_scene,
        }

    def densify_window_sampling(
        *,
        video_path: str,
        evidence_windows: list[EvidenceWindow],
        frame_batches: list[FrameBatch],
        window_ids: list[str] | None = None,
        output_root: str | None = None,
        frames_per_scene: int = 6,
        storyboard_path: str | None = None,
        **_: object,
    ) -> dict[str, object]:
        selected_windows = (
            [window for window in evidence_windows if window.window_id in set(window_ids or [])]
            if window_ids
            else list(evidence_windows)
        )
        window_id_filter = {window.window_id for window in selected_windows}
        scenes = [
            SceneBoundary(
                scene_index=index,
                start_seconds=window.start_time,
                end_seconds=window.end_time,
            )
            for index, window in enumerate(evidence_windows)
            if window.window_id in window_id_filter
        ]
        frame_root = Path(output_root) if output_root is not None else _artifact_root(video_path)
        frame_root.mkdir(parents=True, exist_ok=True)
        densified = sample_frames(
            video_path,
            scenes,
            output_dir=str(frame_root),
            frames_per_scene=frames_per_scene,
        )
        batch_by_scene = {batch.scene_index: batch for batch in frame_batches}
        for batch in densified:
            batch_by_scene[batch.scene_index] = batch
        merged_batches = [batch_by_scene[index] for index in sorted(batch_by_scene)]
        resolved_storyboard = generate_storyboard(
            merged_batches,
            output_path=storyboard_path or str(frame_root / "storyboard.jpg"),
        )
        hydrated_windows = hydrate_evidence_windows(
            evidence_windows,
            merged_batches,
            storyboard_path=resolved_storyboard,
        )
        return {
            "frame_batches": merged_batches,
            "evidence_windows": hydrated_windows,
            "storyboard_path": resolved_storyboard,
            "frames_per_scene": frames_per_scene,
            "window_ids": [window.window_id for window in selected_windows],
        }

    def refine_candidate_cuts(
        *,
        probe: object,
        candidate_cuts: list[CandidateCut],
        frame_batches: list[FrameBatch],
        tracks: list[Sam3EntityTrack],
        label_candidates_by_track: dict[str, list[LabelCandidate]],
        storyboard_path: str | None = None,
    ) -> dict[str, object]:
        merged_candidate_cuts, evidence_windows = build_context_candidate_cuts(
            candidate_cuts=candidate_cuts,
            probe=probe,
            frame_batches=frame_batches,
            tracks=tracks,
            label_candidates_by_track=label_candidates_by_track,
            storyboard_path=storyboard_path,
        )
        return {
            "candidate_cuts": merged_candidate_cuts,
            "evidence_windows": evidence_windows,
        }

    def extract_entities(
        *,
        frame_batches: list[FrameBatch],
        prompts_by_scene: dict[int, list[str]] | None = None,
    ) -> object:
        return tooling_runtime.sam3_client.extract_entities(
            frame_batches, prompts_by_scene=prompts_by_scene
        )

    def recover_with_text_prompt(
        *, frame_batches: list[FrameBatch], text_prompt: str
    ) -> object:
        return tooling_runtime.sam3_client.recover_with_text_prompt(
            frame_batches, text_prompt=text_prompt
        )

    def recover_foreground_sources(
        *,
        frame_batches: list[FrameBatch],
        prompt_vocabulary: list[str] | None = None,
        per_scene_top_k: int = 3,
        score_threshold: float = 0.18,
        **_: object,
    ) -> dict[str, object]:
        prompts_by_scene = _scene_prompt_candidates(
            frame_batches,
            tooling_runtime.label_client,
            prompt_vocabulary=prompt_vocabulary or list(RECOVERY_SCENE_LABELS),
            per_scene_top_k=per_scene_top_k,
            score_threshold=score_threshold,
        )
        if tooling_runtime.runtime_profile == "mig10_safe":
            tooling_runtime.release_client("label")
        track_set = tooling_runtime.sam3_client.extract_entities(
            frame_batches,
            prompts_by_scene=prompts_by_scene,
            score_threshold=0.25,
        )
        track_set.strategy = "scene_prompt_recovery"
        return {
            "track_set": track_set,
            "prompts_by_scene": prompts_by_scene,
        }

    def crop_track_artifacts(
        *,
        frame_batches: list[FrameBatch],
        tracks: list[Sam3EntityTrack],
        output_dir: str,
    ) -> object:
        return crop_tracks(frame_batches, tracks, output_dir=output_dir)

    def embed_track_crops(*, track_image_paths: dict[str, list[str]]) -> object:
        return tooling_runtime.embedding_client.embed_images(track_image_paths)

    def score_track_labels(
        *, track_image_paths: dict[str, list[str]], labels: list[str] | None = None
    ) -> dict[str, list[LabelCandidate]]:
        label_set = labels or list(DEFAULT_TRACK_LABELS)
        return {
            track_id: [
                LabelCandidate(label=score.label, score=round(score.score, 4))
                for score in tooling_runtime.label_client.score_image_labels(
                    image_paths=image_paths, labels=label_set
                )
            ]
            for track_id, image_paths in track_image_paths.items()
        }

    def group_embeddings(
        *,
        embeddings: list[EntityEmbedding],
        tracks_by_id: dict[str, Sam3EntityTrack],
    ) -> object:
        return group_entity_embeddings(embeddings, tracks_by_id=tracks_by_id)

    def routing_priors(*, tracks: list[Sam3EntityTrack]) -> dict[str, object]:
        return {track.track_id: route_track(track) for track in tracks}

    def build_source_semantics(
        *,
        tracks: list[Sam3EntityTrack],
        embeddings: list[EntityEmbedding],
        track_crops: list[TrackCrop],
        label_candidates_by_track: dict[str, list[LabelCandidate]],
        evidence_windows: list[EvidenceWindow],
        candidate_groups: list[CandidateGroup] | None = None,
        routing_decisions_by_track: dict[str, object] | None = None,
    ) -> dict[str, object]:
        tracks_by_id = {track.track_id: track for track in tracks}
        identity_edges = build_identity_edges(
            tracks,
            embeddings,
            label_candidates_by_track=label_candidates_by_track,
        )
        physical_sources = build_provisional_source_tracks(
            tracks,
            identity_edges,
            track_crops=track_crops,
            evidence_windows=evidence_windows,
            candidate_groups=candidate_groups,
            label_candidates_by_track=label_candidates_by_track,
        )
        sound_events = build_sound_event_segments(
            physical_sources,
            tracks_by_id=tracks_by_id,
        )
        ambience_beds = build_ambience_beds(evidence_windows, physical_sources)
        generation_groups = build_generation_groups(
            sound_events,
            ambience_beds,
            physical_sources=physical_sources,
            candidate_groups=candidate_groups,
            routing_decisions_by_track=routing_decisions_by_track,
        )
        return {
            "identity_edges": identity_edges,
            "physical_sources": physical_sources,
            "sound_events": sound_events,
            "ambience_beds": ambience_beds,
            "generation_groups": generation_groups,
        }

    def validator(*, bundle: MultitrackDescriptionBundle) -> object:
        return validate_bundle(bundle)

    def rerun_description_writer(
        *,
        generation_groups: list[object],
        sound_events: list[object],
        ambience_beds: list[object],
        physical_sources: list[object],
    ) -> object:
        return synthesize_canonical_descriptions(
            list(generation_groups),
            sound_events=list(sound_events),
            ambience_beds=list(ambience_beds),
            physical_sources=list(physical_sources),
            description_writer=getattr(tooling_runtime, "description_writer", None),
        )

    return {
        "structural_overview": ToolDefinition(
            "structural_overview",
            structural_overview,
            "Probe video and build candidate cuts/evidence windows.",
        ),
        "extract_entities": ToolDefinition(
            "extract_entities",
            extract_entities,
            "Run baseline prompt-seeded SAM extraction.",
        ),
        "densify_window_sampling": ToolDefinition(
            "densify_window_sampling",
            densify_window_sampling,
            "Resample problematic windows with denser frame coverage.",
        ),
        "refine_candidate_cuts": ToolDefinition(
            "refine_candidate_cuts",
            refine_candidate_cuts,
            "Merge structural, source-lifecycle, label-context, and interaction cut cues.",
        ),
        "recover_with_text_prompt": ToolDefinition(
            "recover_with_text_prompt",
            recover_with_text_prompt,
            "Run manual recovery-only extraction.",
        ),
        "recover_foreground_sources": ToolDefinition(
            "recover_foreground_sources",
            recover_foreground_sources,
            "Recover missing foreground sources with scene-specific prompt expansion.",
        ),
        "crop_tracks": ToolDefinition(
            "crop_tracks",
            crop_track_artifacts,
            "Generate crop artifacts from track geometry.",
        ),
        "embed_track_crops": ToolDefinition(
            "embed_track_crops",
            embed_track_crops,
            "Embed track crops with the embedding model.",
        ),
        "score_track_labels": ToolDefinition(
            "score_track_labels",
            score_track_labels,
            "Score labels against crop-backed track evidence.",
        ),
        "group_embeddings": ToolDefinition(
            "group_embeddings",
            group_embeddings,
            "Build grouping proposals from embeddings.",
        ),
        "routing_priors": ToolDefinition(
            "routing_priors", routing_priors, "Compute deterministic routing priors."
        ),
        "build_source_semantics": ToolDefinition(
            "build_source_semantics",
            build_source_semantics,
            "Build source/event/generation semantic layers.",
        ),
        "validate_bundle": ToolDefinition(
            "validate_bundle", validator, "Run typed bundle validators."
        ),
        "rerun_description_writer": ToolDefinition(
            "rerun_description_writer",
            rerun_description_writer,
            "Rewrite canonical descriptions from the current structured bundle state.",
        ),
    }


def _scene_prompt_candidates(
    frame_batches: list[FrameBatch],
    label_client: object,
    *,
    prompt_vocabulary: list[str],
    per_scene_top_k: int,
    score_threshold: float,
) -> dict[int, list[str]]:
    prompts_by_scene: dict[int, list[str]] = {}
    for batch in frame_batches:
        image_paths = [frame.image_path for frame in batch.frames]
        if not image_paths:
            continue
        scored = label_client.score_image_labels(
            image_paths=image_paths,
            labels=prompt_vocabulary,
        )
        prompts = [
            score.label
            for score in scored[: max(per_scene_top_k, 1)]
            if score.score >= score_threshold
        ]
        prompts_by_scene[batch.scene_index] = prompts or list(prompt_vocabulary[:3])
    return prompts_by_scene
