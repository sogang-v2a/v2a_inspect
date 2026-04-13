from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
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
    create_silent_analysis_video,
    evidence_windows_to_scene_boundaries,
    export_window_clips,
    generate_storyboard,
    group_entity_embeddings,
    hydrate_evidence_windows,
    probe_video,
    sample_frames,
)
from v2a_inspect.tools.types import CandidateGroup, EntityEmbedding, FrameBatch, Sam3EntityTrack, SceneBoundary

from .crops import crop_tracks
from .descriptions import synthesize_canonical_descriptions
from .gemini_grouping import group_generation_groups
from .gemini_proposal_grounding import GroundedWindowProposal
from .gemini_routing import route_generation_groups
from .gemini_source_proposal import WindowSourceProposal
from .gemini_source_semantics import build_source_and_event_semantics
from .reid import build_identity_edges, build_provisional_source_tracks
from .scene_hypotheses import RegionProposal, propose_moving_regions
from .validators import validate_bundle

if TYPE_CHECKING:
    from .runtime import ToolingRuntime


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    handler: Callable[..., object]
    description: str


def _artifact_root(video_path: str) -> Path:
    from .settings import get_server_runtime_settings

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


def build_tool_registry(tooling_runtime: "ToolingRuntime") -> dict[str, ToolDefinition]:
    def structural_overview(
        *,
        video_path: str,
        target_scene_seconds: float = 5.0,
        frames_per_scene: int = 2,
        output_root: str | None = None,
    ) -> dict[str, object]:
        frame_root = Path(output_root) if output_root is not None else _artifact_root(video_path)
        frame_root.mkdir(parents=True, exist_ok=True)
        analysis_video_path = create_silent_analysis_video(
            video_path,
            output_path=str(frame_root / "analysis_silent.mp4"),
        )
        probe = probe_video(analysis_video_path)
        candidate_cuts = build_candidate_cuts(
            analysis_video_path,
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
        frame_batches = sample_frames(
            analysis_video_path,
            scenes,
            output_dir=str(frame_root),
            frames_per_scene=frames_per_scene,
        )
        storyboard_path = generate_storyboard(
            frame_batches,
            output_path=str(frame_root / "storyboard.jpg"),
        )
        clip_paths_by_window = export_window_clips(
            analysis_video_path,
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
            "analysis_video_path": analysis_video_path,
        }

    def propose_source_hypotheses(
        *,
        frame_batches: list[FrameBatch],
        storyboard_path: str,
        output_root: str,
    ) -> dict[str, object]:
        moving_regions = propose_moving_regions(
            frame_batches,
            output_root=str(Path(output_root) / "motion_regions"),
        )
        proposer = getattr(tooling_runtime, "source_proposer", None)
        proposals = (
            proposer.propose(
                frame_batches=frame_batches,
                storyboard_path=storyboard_path,
                moving_regions_by_scene=moving_regions,
            )
            if proposer is not None
            else {}
        )
        return {
            "scene_hypotheses_by_window": {
                scene_index: proposal.model_dump(mode="json")
                for scene_index, proposal in proposals.items()
            },
            "moving_regions_by_window": {
                scene_index: [proposal.model_dump(mode="json") for proposal in proposals_by_scene]
                for scene_index, proposals_by_scene in moving_regions.items()
            },
            "proposal_provenance_by_window": {
                scene_index: {
                    "motion_region_count": len(moving_regions.get(scene_index, [])),
                    "source_proposal": proposals[scene_index].model_dump(mode="json"),
                }
                for scene_index in proposals
            },
            "storyboard_path": storyboard_path,
        }

    def verify_scene_hypotheses(
        *,
        frame_batches: list[FrameBatch],
        scene_hypotheses_by_window: dict[int, dict[str, object]],
        moving_regions_by_window: dict[int, list[dict[str, object]]],
        storyboard_path: str | None = None,
    ) -> dict[str, object]:
        grounder = getattr(tooling_runtime, "proposal_grounder", None)
        proposals = {
            scene_index: WindowSourceProposal.model_validate(payload)
            for scene_index, payload in scene_hypotheses_by_window.items()
        }
        moving_regions = {
            scene_index: [RegionProposal.model_validate(item) for item in payload]
            for scene_index, payload in moving_regions_by_window.items()
        }
        grounded = (
            grounder.ground(
                frame_batches=frame_batches,
                storyboard_path=storyboard_path,
                proposals_by_scene=proposals,
                moving_regions_by_scene=moving_regions,
                label_client=tooling_runtime.label_client,
            )
            if grounder is not None
            else {
                scene_index: GroundedWindowProposal(
                    extraction_prompts=[],
                    semantic_hints=[],
                    rejected_phrases=[],
                    unresolved_phrases=_proposal_phrases(proposal),
                    rationale="proposal grounding unavailable",
                )
                for scene_index, proposal in proposals.items()
            }
        )
        return {
            "verified_hypotheses_by_window": {
                scene_index: payload.model_dump(mode="json")
                for scene_index, payload in grounded.items()
            },
            "prompts_by_scene": {
                scene_index: payload.extraction_prompts
                for scene_index, payload in grounded.items()
            },
            "semantic_hints_by_window": {
                scene_index: payload.semantic_hints
                for scene_index, payload in grounded.items()
            },
            "proposal_provenance_by_window": {
                scene_index: {
                    "grounded_proposal": payload.model_dump(mode="json"),
                }
                for scene_index, payload in grounded.items()
            },
        }

    def densify_window_sampling(
        *,
        analysis_video_path: str,
        evidence_windows: list[EvidenceWindow],
        frame_batches: list[FrameBatch],
        window_ids: list[str] | None = None,
        output_root: str | None = None,
        frames_per_scene: int = 4,
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
        frame_root = Path(output_root) if output_root is not None else _artifact_root(analysis_video_path)
        frame_root.mkdir(parents=True, exist_ok=True)
        densified = sample_frames(
            analysis_video_path,
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
        del tracks, label_candidates_by_track
        merged_candidate_cuts, evidence_windows = build_context_candidate_cuts(
            candidate_cuts=candidate_cuts,
            probe=probe,
            frame_batches=frame_batches,
            tracks=[],
            label_candidates_by_track={},
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
        storyboard_path: str | None = None,
        output_root: str | None = None,
        **_: object,
    ) -> object:
        resolved_prompts = prompts_by_scene
        if resolved_prompts is None:
            proposals = propose_source_hypotheses(
                frame_batches=frame_batches,
                storyboard_path=storyboard_path or "",
                output_root=output_root or tempfile.mkdtemp(prefix="v2a_prompt_refresh_"),
            )
            verified = verify_scene_hypotheses(
                frame_batches=frame_batches,
                scene_hypotheses_by_window=proposals["scene_hypotheses_by_window"],
                moving_regions_by_window=proposals["moving_regions_by_window"],
                storyboard_path=storyboard_path,
            )
            resolved_prompts = dict(verified["prompts_by_scene"])
        return tooling_runtime.sam3_client.extract_entities(
            frame_batches,
            prompts_by_scene=resolved_prompts,
            score_threshold=0.25,
        )

    def recover_foreground_sources(
        *,
        frame_batches: list[FrameBatch],
        storyboard_path: str | None = None,
        output_root: str | None = None,
        prompt_vocabulary: list[str] | None = None,
        **_: object,
    ) -> dict[str, object]:
        del prompt_vocabulary
        refreshed = propose_source_hypotheses(
            frame_batches=frame_batches,
            storyboard_path=storyboard_path or "",
            output_root=output_root or tempfile.mkdtemp(prefix="v2a_prompt_recovery_"),
        )
        verified = verify_scene_hypotheses(
            frame_batches=frame_batches,
            scene_hypotheses_by_window=refreshed["scene_hypotheses_by_window"],
            moving_regions_by_window=refreshed["moving_regions_by_window"],
            storyboard_path=storyboard_path,
        )
        track_set = tooling_runtime.sam3_client.extract_entities(
            frame_batches,
            prompts_by_scene=dict(verified["prompts_by_scene"]),
            score_threshold=0.22,
        )
        track_set.strategy = "scene_prompt_recovery"
        return {
            "track_set": track_set,
            **refreshed,
            **verified,
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
        label_set = _dedupe(labels or [])
        if not label_set:
            return {}
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

    def build_source_semantics(
        *,
        tracks: list[Sam3EntityTrack],
        embeddings: list[EntityEmbedding],
        track_crops: list[TrackCrop],
        label_candidates_by_track: dict[str, list[LabelCandidate]],
        evidence_windows: list[EvidenceWindow],
        candidate_groups: list[CandidateGroup] | None = None,
        **_: object,
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
        semantic_started = perf_counter()
        semantic_outputs = build_source_and_event_semantics(
            physical_sources=physical_sources,
            tracks_by_id=tracks_by_id,
            track_crops=track_crops,
            evidence_windows=evidence_windows,
            interpreter=getattr(tooling_runtime, "source_semantics_interpreter", None),
        )
        generation_groups = group_generation_groups(
            sound_events=semantic_outputs["sound_events"],
            ambience_beds=semantic_outputs["ambience_beds"],
            physical_sources=semantic_outputs["physical_sources"],
            grouping_judge=getattr(tooling_runtime, "grouping_judge", None),
        )
        generation_groups = route_generation_groups(
            generation_groups=generation_groups,
            sound_events=semantic_outputs["sound_events"],
            ambience_beds=semantic_outputs["ambience_beds"],
            physical_sources=semantic_outputs["physical_sources"],
            routing_judge=getattr(tooling_runtime, "routing_judge", None),
        )
        return {
            "identity_edges": identity_edges,
            "physical_sources": semantic_outputs["physical_sources"],
            "sound_events": semantic_outputs["sound_events"],
            "ambience_beds": semantic_outputs["ambience_beds"],
            "generation_groups": generation_groups,
            "recipe_signatures": {},
            "recipe_grouping_seconds": round(perf_counter() - semantic_started, 4),
        }

    def group_acoustic_recipe_semantics(
        *,
        sound_events: list[object],
        ambience_beds: list[object],
        physical_sources: list[object],
        **_: object,
    ) -> dict[str, object]:
        groups = group_generation_groups(
            sound_events=list(sound_events),
            ambience_beds=list(ambience_beds),
            physical_sources=list(physical_sources),
            grouping_judge=getattr(tooling_runtime, "grouping_judge", None),
        )
        routed = route_generation_groups(
            generation_groups=groups,
            sound_events=list(sound_events),
            ambience_beds=list(ambience_beds),
            physical_sources=list(physical_sources),
            routing_judge=getattr(tooling_runtime, "routing_judge", None),
        )
        return {"generation_groups": routed, "recipe_signatures": {}}

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
            "Probe video, create a silent analysis copy, and build structural evidence windows.",
        ),
        "propose_source_hypotheses": ToolDefinition(
            "propose_source_hypotheses",
            propose_source_hypotheses,
            "Use Gemini to propose open-world visible sources from frames, storyboard, and motion crops.",
        ),
        "verify_scene_hypotheses": ToolDefinition(
            "verify_scene_hypotheses",
            verify_scene_hypotheses,
            "Ground Gemini source proposals into extraction prompts and semantic hints.",
        ),
        "extract_entities": ToolDefinition(
            "extract_entities",
            extract_entities,
            "Run SAM3 extraction from grounded prompts only.",
        ),
        "densify_window_sampling": ToolDefinition(
            "densify_window_sampling",
            densify_window_sampling,
            "Resample problematic windows with denser frame coverage.",
        ),
        "refine_candidate_cuts": ToolDefinition(
            "refine_candidate_cuts",
            refine_candidate_cuts,
            "Hydrate evidence windows after structural cut refinement.",
        ),
        "recover_foreground_sources": ToolDefinition(
            "recover_foreground_sources",
            recover_foreground_sources,
            "Refresh open-world Gemini proposals and retry extraction.",
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
            "Score dynamic Gemini-derived labels against crop-backed track evidence.",
        ),
        "group_embeddings": ToolDefinition(
            "group_embeddings",
            group_embeddings,
            "Build deterministic grouping proposals from embeddings.",
        ),
        "build_source_semantics": ToolDefinition(
            "build_source_semantics",
            build_source_semantics,
            "Interpret sources/events with Gemini and build generation groups plus routing.",
        ),
        "group_acoustic_recipes": ToolDefinition(
            "group_acoustic_recipes",
            group_acoustic_recipe_semantics,
            "Regroup existing events/ambience with Gemini semantic merge judgments.",
        ),
        "validate_bundle": ToolDefinition(
            "validate_bundle",
            validator,
            "Run bundle validators.",
        ),
        "rerun_description_writer": ToolDefinition(
            "rerun_description_writer",
            rerun_description_writer,
            "Rewrite canonical descriptions with Gemini.",
        ),
    }


def _proposal_phrases(proposal: WindowSourceProposal) -> list[str]:
    return _dedupe(
        [
            *proposal.visible_sources,
            *proposal.background_sources,
            *proposal.interactions,
            *proposal.materials_surfaces,
            *proposal.uncertain_regions,
        ]
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
