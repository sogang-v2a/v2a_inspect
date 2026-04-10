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
from v2a_inspect.tools.types import CandidateGroup, EntityEmbedding, FrameBatch, Sam3EntityTrack

from .constants import DEFAULT_TRACK_LABELS
from .crops import crop_tracks
from .settings import get_server_runtime_settings
from .reid import build_identity_edges, build_provisional_source_tracks
from .semantics import (
    build_ambience_beds,
    build_generation_groups,
    build_sound_event_segments,
)
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
            frames_per_scene=2,
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

    return {
        "structural_overview": ToolDefinition(
            "structural_overview",
            structural_overview,
            "Probe video and build candidate cuts/evidence windows.",
        ),
        "extract_entities": ToolDefinition(
            "extract_entities", extract_entities, "Run prompt-free SAM extraction."
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
    }
