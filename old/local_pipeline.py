from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from v2a_inspect.contracts import (
    AmbienceBed,
    CandidateCut,
    EvidenceWindow,
    IdentityEdge,
    LabelCandidate,
    MultitrackDescriptionBundle,
    PhysicalSourceTrack,
    SoundEventSegment,
    TrackCrop,
)
from v2a_inspect.review import persist_bundle
from v2a_inspect.settings_views import get_client_runtime_settings
from v2a_inspect.tools.types import (
    EntityEmbedding,
    FrameBatch,
    Sam3RegionSeed,
    Sam3TrackSet,
    VideoProbe,
)
from v2a_inspect.workflows import InspectOptions, InspectState

from .agentic import run_agentic_tool_loop
from .crops import group_crop_paths_by_track
from .finalize import build_final_bundle, build_interim_bundle
from .label_vocabulary import dynamic_label_vocabulary
from .local_runtime import LocalToolingRuntime, RemoteRuntimeSnapshot
from .telemetry import ensure_runtime_trace_path, record_stage, stage_start
from .tool_registry import build_tool_registry


def run_local_inspect_raw(
    *,
    video_path: str,
    options: InspectOptions,
) -> dict[str, object]:
    server_base_url = options.server_base_url or get_client_runtime_settings().server_base_url
    tooling_runtime = LocalToolingRuntime(
        server_base_url=server_base_url,
        remote_timeout_seconds=float(options.remote_timeout_seconds),
        semantic_model=options.gemini_model,
    )
    if options.pipeline_mode == "agentic_tool_first":
        state = _run_agentic_tool_first_pipeline(
            video_path=video_path,
            options=options,
            tooling_runtime=tooling_runtime,
        )
    else:
        state = _run_tool_first_pipeline(
            video_path=video_path,
            options=options,
            tooling_runtime=tooling_runtime,
        )
    bundle = state["multitrack_bundle"]
    after = _runtime_snapshot_or_fallback(tooling_runtime, options)
    bundle.pipeline_metadata["effective_runtime_profile"] = after.effective_runtime_profile
    bundle.pipeline_metadata["runtime_profile_source"] = after.runtime_profile_source
    bundle.pipeline_metadata["runtime_residency_mode"] = after.residency_mode
    bundle.pipeline_metadata["warm_start"] = state.get("warm_start", False)
    bundle.pipeline_metadata["resident_models_before_run"] = list(state.get("resident_models_before_run", []))
    bundle.pipeline_metadata["resident_models_after_run"] = after.resident_models
    _persist_runtime_bundle(bundle, state)
    return {
        "multitrack_bundle": bundle.model_dump(mode="json"),
        "warnings": list(state.get("warnings", [])),
        "progress_messages": list(state.get("progress_messages", [])),
        "effective_runtime_profile": after.effective_runtime_profile,
        "runtime_profile_source": after.runtime_profile_source,
        "residency_mode": after.residency_mode,
        "warm_start": state.get("warm_start", False),
        "resident_models_before_run": list(state.get("resident_models_before_run", [])),
        "resident_models_after_run": after.resident_models,
        "server_base_url": server_base_url,
    }


def _run_tool_first_pipeline(
    *,
    video_path: str,
    options: InspectOptions,
    tooling_runtime: LocalToolingRuntime,
    bundle_mode: str = "final",
) -> InspectState:
    registry = build_tool_registry(tooling_runtime)
    snapshot = _runtime_snapshot_or_fallback(tooling_runtime, options)
    state: InspectState = {
        "video_path": video_path,
        "options": options,
        "recovery_actions": [],
        "recovery_attempts": [],
        "stage_history": [],
        "effective_runtime_profile": snapshot.effective_runtime_profile or options.runtime_profile,
        "runtime_profile_source": snapshot.runtime_profile_source or "client_options",
        "runtime_residency_mode": snapshot.residency_mode or tooling_runtime.residency_mode,
        "resident_models_before_run": list(snapshot.resident_models),
        "warm_start": set(("sam3", "embedding", "label")).issubset(set(snapshot.resident_models)),
    }
    started = stage_start()
    structural = _mapping_result(
        registry["structural_overview"].handler(
            video_path=video_path,
            target_scene_seconds=5.0,
        )
    )
    probe = cast(VideoProbe, structural["probe"])
    candidate_cuts = list(cast(list[CandidateCut], structural["candidate_cuts"]))
    evidence_windows = list(cast(list[EvidenceWindow], structural["evidence_windows"]))
    frame_batches = list(cast(list[FrameBatch], structural["frame_batches"]))
    storyboard_path = str(structural["storyboard_path"])
    artifact_run_dir = str(structural["artifact_root"])
    analysis_video_path = str(structural["analysis_video_path"])
    state["artifact_run_dir"] = artifact_run_dir
    state["analysis_video_path"] = analysis_video_path
    state["storyboard_path"] = storyboard_path
    state["frames_per_window"] = _int_value(structural.get("frames_per_scene"), 2)
    ensure_runtime_trace_path(state)
    record_stage(
        state,
        stage="structural_overview",
        started_at=started,
        metrics={
            "candidate_cut_count": len(candidate_cuts),
            "evidence_window_count": len(evidence_windows),
            "sampled_frame_count": sum(len(batch.frames) for batch in frame_batches),
            "frames_per_window": state["frames_per_window"],
        },
    )

    started = stage_start()
    source_hypotheses = _mapping_result(
        registry["propose_source_hypotheses"].handler(
            frame_batches=frame_batches,
            storyboard_path=storyboard_path,
            output_root=artifact_run_dir,
        )
    )
    state["scene_hypotheses_by_window"] = cast(
        dict[int, dict[str, object]],
        source_hypotheses["scene_hypotheses_by_window"],
    )
    state["proposal_provenance_by_window"] = cast(
        dict[int, dict[str, object]],
        source_hypotheses["proposal_provenance_by_window"],
    )
    moving_regions_by_window = cast(
        dict[int, list[dict[str, object]]],
        source_hypotheses.get("moving_regions_by_window", {}),
    )
    source_warnings = source_hypotheses.get("warnings")
    if isinstance(source_warnings, list) and source_warnings:
        state.setdefault("warnings", []).extend([str(item) for item in source_warnings if item])
    record_stage(
        state,
        stage="propose_source_hypotheses",
        started_at=started,
        metrics={
            "window_count": len(frame_batches),
            "motion_region_count": sum(len(payload) for payload in moving_regions_by_window.values()),
            "window_hypothesis_count": len(state["scene_hypotheses_by_window"]),
        },
    )

    started = stage_start()
    verified_hypotheses = _mapping_result(
        registry["verify_scene_hypotheses"].handler(
            frame_batches=frame_batches,
            scene_hypotheses_by_window=source_hypotheses["scene_hypotheses_by_window"],
            moving_regions_by_window=source_hypotheses["moving_regions_by_window"],
            storyboard_path=storyboard_path,
        )
    )
    state["verified_hypotheses_by_window"] = cast(
        dict[int, dict[str, object]],
        verified_hypotheses["verified_hypotheses_by_window"],
    )
    state["scene_prompt_candidates"] = cast(
        dict[int, list[str]],
        verified_hypotheses["prompts_by_scene"],
    )
    state["region_seeds_by_scene"] = {
        scene_index: [Sam3RegionSeed.model_validate(seed) for seed in seeds]
        for scene_index, seeds in cast(dict[int, list[dict[str, object]]], verified_hypotheses["region_seeds_by_scene"]).items()
    }
    verified_provenance = cast(
        dict[int, dict[str, object]],
        verified_hypotheses["proposal_provenance_by_window"],
    )
    state["proposal_provenance_by_window"] = {
        scene_index: {
            **dict(state.get("proposal_provenance_by_window", {})).get(scene_index, {}),
            **payload,
        }
        for scene_index, payload in verified_provenance.items()
    }
    verified_warnings = verified_hypotheses.get("warnings")
    if isinstance(verified_warnings, list) and verified_warnings:
        state.setdefault("warnings", []).extend([str(item) for item in verified_warnings if item])
    record_stage(
        state,
        stage="verify_scene_hypotheses",
        started_at=started,
        metrics={
            "verified_window_count": len(state["verified_hypotheses_by_window"]),
            "verified_prompt_count": sum(len(prompts) for prompts in state["scene_prompt_candidates"].values()),
            "region_seed_count": sum(len(seeds) for seeds in state["region_seeds_by_scene"].values()),
            "uncertain_hypothesis_count": sum(_list_length(payload.get("unresolved_phrases")) for payload in state["verified_hypotheses_by_window"].values()),
        },
    )

    started = stage_start()
    extraction = cast(
        Sam3TrackSet,
        registry["extract_entities"].handler(
            frame_batches=frame_batches,
            prompts_by_scene={scene_index: list(prompts) for scene_index, prompts in dict(state["scene_prompt_candidates"]).items()},
            region_seeds_by_scene={scene_index: [seed.model_dump(mode="json") for seed in seeds] for scene_index, seeds in dict(state["region_seeds_by_scene"]).items()},
            storyboard_path=storyboard_path,
            output_root=artifact_run_dir,
        ),
    )
    tracks = _coerce_tracks(extraction)
    record_stage(
        state,
        stage="extract_entities",
        started_at=started,
        metrics={"track_count": len(tracks), "extraction_strategy": getattr(extraction, "strategy", None)},
    )

    started = stage_start()
    track_crops = cast(
        list[TrackCrop],
        registry["crop_tracks"].handler(
            frame_batches=frame_batches,
            tracks=tracks,
            output_dir=str(Path(storyboard_path).parent / "crops"),
        ),
    )
    track_image_paths = group_crop_paths_by_track(track_crops)
    record_stage(
        state,
        stage="crop_tracks",
        started_at=started,
        metrics={"crop_count": len(track_crops), "tracks_with_crops": len(track_image_paths)},
    )

    started = stage_start()
    embeddings = cast(list[EntityEmbedding], registry["embed_track_crops"].handler(track_image_paths=track_image_paths)) if track_image_paths else []
    candidate_groups = list(
        getattr(
            registry["group_embeddings"].handler(
                embeddings=embeddings,
                tracks_by_id={track.track_id: track for track in tracks},
            ),
            "groups",
            [],
        )
    )
    record_stage(
        state,
        stage="embed_track_crops",
        started_at=started,
        metrics={"embedding_count": len(embeddings), "candidate_group_count": len(candidate_groups)},
    )

    started = stage_start()
    track_label_candidates = (
        cast(
            dict[str, list[LabelCandidate]],
            registry["score_track_labels"].handler(
                track_image_paths=track_image_paths,
                labels=dynamic_label_vocabulary(
                    dict(state.get("verified_hypotheses_by_window", {})),
                    dict(state.get("scene_hypotheses_by_window", {})),
                ),
            ),
        )
        if track_image_paths
        else {}
    )
    record_stage(
        state,
        stage="score_track_labels",
        started_at=started,
        metrics={"labeled_track_count": len(track_label_candidates), "routing_track_count": 0},
    )

    started = stage_start()
    refined_structure = _mapping_result(
        registry["refine_candidate_cuts"].handler(
            probe=probe,
            candidate_cuts=candidate_cuts,
            frame_batches=frame_batches,
            tracks=tracks,
            label_candidates_by_track=track_label_candidates,
            storyboard_path=storyboard_path,
        )
    )
    candidate_cuts = list(cast(list[CandidateCut], refined_structure["candidate_cuts"]))
    evidence_windows = list(cast(list[EvidenceWindow], refined_structure["evidence_windows"]))
    record_stage(
        state,
        stage="refine_candidate_cuts",
        started_at=started,
        metrics={"candidate_cut_count": len(candidate_cuts), "evidence_window_count": len(evidence_windows)},
    )

    started = stage_start()
    semantics = _mapping_result(
        registry["build_source_semantics"].handler(
            tracks=tracks,
            embeddings=embeddings,
            track_crops=track_crops,
            label_candidates_by_track=track_label_candidates,
            evidence_windows=evidence_windows,
            candidate_groups=candidate_groups,
        )
    )
    physical_sources = list(cast(list[PhysicalSourceTrack], semantics["physical_sources"]))
    sound_events = list(cast(list[SoundEventSegment], semantics["sound_events"]))
    ambience_beds = list(cast(list[AmbienceBed], semantics["ambience_beds"]))
    generation_groups = list(cast(list[object], semantics["generation_groups"]))
    recipe_signatures = _mapping_result(semantics.get("recipe_signatures", {}))
    identity_edges = list(cast(list[IdentityEdge], semantics["identity_edges"]))
    record_stage(
        state,
        stage="build_source_semantics",
        started_at=started,
        metrics={
            "identity_edge_count": len(identity_edges),
            "physical_source_count": len(physical_sources),
            "sound_event_count": len(sound_events),
            "ambience_bed_count": len(ambience_beds),
            "generation_group_count": len(generation_groups),
            "recipe_signature_count": len(recipe_signatures),
            "recipe_grouping_seconds": semantics.get("recipe_grouping_seconds"),
        },
    )

    state.update({
        "artifact_run_dir": artifact_run_dir,
        "video_probe": probe,
        "candidate_cuts": candidate_cuts,
        "evidence_windows": evidence_windows,
        "frame_batches": frame_batches,
        "analysis_video_path": analysis_video_path,
        "storyboard_path": storyboard_path,
        "sam3_track_set": extraction,
        "scene_prompt_candidates": dict(state.get("scene_prompt_candidates", {})),
        "region_seeds_by_scene": dict(state.get("region_seeds_by_scene", {})),
        "scene_hypotheses_by_window": dict(state.get("scene_hypotheses_by_window", {})),
        "proposal_provenance_by_window": dict(state.get("proposal_provenance_by_window", {})),
        "verified_hypotheses_by_window": dict(state.get("verified_hypotheses_by_window", {})),
        "track_crops": track_crops,
        "entity_embeddings": embeddings,
        "candidate_groups": candidate_groups,
        "track_routing_decisions": {},
        "track_label_candidates": track_label_candidates,
        "physical_sources": physical_sources,
        "sound_event_segments": sound_events,
        "ambience_beds": ambience_beds,
        "generation_groups": generation_groups,
        "recipe_signatures_by_group": {key: _jsonable_value(value) for key, value in recipe_signatures.items()},
        "identity_edges": identity_edges,
        "warnings": list(state.get("warnings", [])),
        "errors": [],
        "progress_messages": [
            f"Tool-first pipeline: proposed {len(candidate_cuts)} candidate cuts.",
            f"Tool-first pipeline: built {len(evidence_windows)} evidence windows.",
            f"Tool-first pipeline: sampled {sum(len(batch.frames) for batch in frame_batches)} frames.",
            f"Tool-first pipeline: proposed {sum(len(prompts) for prompts in state.get('scene_prompt_candidates', {}).values())} extraction prompts.",
            f"Tool-first pipeline: prepared {sum(len(seeds) for seeds in state.get('region_seeds_by_scene', {}).values())} region-grounded SAM seeds.",
            f"Tool-first pipeline: extracted {len(tracks)} source tracks.",
            f"Tool-first pipeline: generated {len(track_crops)} track crops.",
            f"Tool-first pipeline: embedded {len(embeddings)} crop-backed track identities.",
            f"Tool-first pipeline: refined structure with {len(candidate_cuts)} merged candidate cuts.",
            "Tool-first pipeline: built source, event, ambience, and generation-group semantics.",
        ],
    })
    bundle = build_final_bundle(state, description_writer=tooling_runtime.description_writer) if bundle_mode == "final" else build_interim_bundle(state)
    started = stage_start()
    _persist_runtime_bundle(bundle, state)
    record_stage(
        state,
        stage="persist_bundle",
        started_at=started,
        metrics={"bundle_path": state.get("bundle_path"), "pipeline_mode": options.pipeline_mode},
    )
    state["multitrack_bundle"] = bundle
    return state


def _run_agentic_tool_first_pipeline(
    *,
    video_path: str,
    options: InspectOptions,
    tooling_runtime: LocalToolingRuntime,
) -> InspectState:
    state = _run_tool_first_pipeline(
        video_path=video_path,
        options=options.model_copy(update={"pipeline_mode": "tool_first_foundation"}),
        tooling_runtime=tooling_runtime,
        bundle_mode="interim",
    )
    state["options"] = options
    state, planner_state, trace_path = run_agentic_tool_loop(
        inspect_state=state,
        tooling_runtime=tooling_runtime,
    )
    state["agent_trace_path"] = trace_path
    bundle = state["multitrack_bundle"]
    bundle.artifacts.trace_path = trace_path
    bundle.pipeline_metadata["agent_review_trace_path"] = trace_path
    bundle.pipeline_metadata["agent_review_issue_count"] = len(planner_state.issues)
    bundle.pipeline_metadata["agent_review_tool_calls"] = len(planner_state.tool_calls)
    _persist_runtime_bundle(bundle, state)
    state["multitrack_bundle"] = bundle
    return state


def _runtime_snapshot_or_fallback(
    tooling_runtime: LocalToolingRuntime,
    options: InspectOptions,
) -> RemoteRuntimeSnapshot:
    try:
        snapshot = tooling_runtime.remote_runtime_snapshot()
    except Exception as exc:  # noqa: BLE001
        return RemoteRuntimeSnapshot(
            effective_runtime_profile=options.runtime_profile,
            runtime_profile_source=f"client_fallback:{type(exc).__name__}",
            residency_mode=tooling_runtime.residency_mode,
            resident_models=[],
        )
    return RemoteRuntimeSnapshot(
        effective_runtime_profile=snapshot.effective_runtime_profile or options.runtime_profile,
        runtime_profile_source=snapshot.runtime_profile_source or "remote_runtime_info",
        residency_mode=snapshot.residency_mode or tooling_runtime.residency_mode,
        resident_models=list(snapshot.resident_models),
    )


def _coerce_tracks(extraction_result: object) -> list[object]:
    if isinstance(extraction_result, dict):
        extraction_payload = cast(dict[str, object], extraction_result)
        tracks = extraction_payload.get("tracks")
    else:
        tracks = getattr(extraction_result, "tracks", [])
    if isinstance(tracks, list):
        return list(tracks)
    return list(tracks) if tracks is not None else []


def _mapping_result(result: object) -> dict[str, object]:
    if not isinstance(result, Mapping):
        raise TypeError(f"Expected mapping-like tool result, received {type(result).__name__}.")
    return {str(key): value for key, value in result.items()}


def _int_value(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _list_length(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _jsonable_value(value: object) -> dict[str, object]:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return cast(dict[str, object], model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {"value": str(value)}


def _persist_runtime_bundle(bundle: MultitrackDescriptionBundle, state: InspectState) -> None:
    artifact_run_dir = state.get("artifact_run_dir")
    if not isinstance(artifact_run_dir, str) or not artifact_run_dir:
        return
    bundle_path = Path(artifact_run_dir) / "bundle.json"
    state["bundle_path"] = str(bundle_path)
    bundle.artifacts.bundle_path = str(bundle_path)
    persist_bundle(bundle, bundle_path)
