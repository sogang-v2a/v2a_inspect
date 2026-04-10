from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from v2a_inspect.agent import (
    AgentIssue,
    PlannerState,
    ToolExecutor,
    mark_issue_attempted,
    plan_next_action,
    resolve_issue,
)
from v2a_inspect.contracts import MultitrackDescriptionBundle, PhysicalSourceTrack
from v2a_inspect.contracts.adapters import bundle_to_grouped_analysis
from v2a_inspect.workflows import InspectState

from .crops import group_crop_paths_by_track
from .finalize import build_final_bundle
from .tool_registry import build_tool_registry

if TYPE_CHECKING:
    from .runtime import ToolingRuntime


def run_agentic_tool_loop(
    *,
    inspect_state: InspectState,
    tooling_runtime: "ToolingRuntime",
    max_actions: int = 3,
) -> tuple[InspectState, PlannerState, str]:
    registry = {
        name: definition.handler
        for name, definition in build_tool_registry(tooling_runtime).items()
    }
    bundle = inspect_state.get("multitrack_bundle") or build_final_bundle(inspect_state)
    trace_path = _trace_path(bundle)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.touch(exist_ok=True)

    planner_state = PlannerState(
        video_id=bundle.video_id,
        issues=_build_issues(bundle=bundle, inspect_state=inspect_state),
    )
    executor = ToolExecutor(registry=registry, trace_path=trace_path)

    actions_run = 0
    while actions_run < max_actions:
        action = plan_next_action(planner_state)
        if action is None:
            break
        planner_state = mark_issue_attempted(planner_state, action.issue_id)
        planner_state, result = executor.execute(planner_state, action)
        inspect_state = _apply_action_result(
            inspect_state=inspect_state,
            tooling_runtime=tooling_runtime,
            tool_name=action.tool_name,
            result=result,
            registry=registry,
        )
        bundle = build_final_bundle(inspect_state)
        inspect_state["multitrack_bundle"] = bundle
        inspect_state["grouped_analysis"] = bundle_to_grouped_analysis(bundle)
        inspect_state["scene_analysis"] = inspect_state["grouped_analysis"].scene_analysis

        current_issue_ids = {
            issue.issue_id
            for issue in _build_issues(bundle=bundle, inspect_state=inspect_state)
        }
        decision = "accept" if action.issue_id not in current_issue_ids else "retry"
        planner_state = executor.record_decision(
            planner_state,
            issue_id=action.issue_id,
            decision=decision,
            rationale=f"{action.tool_name} completed during agentic tool loop",
            confidence=0.8 if decision == "accept" else 0.45,
        )
        if decision == "accept":
            planner_state = resolve_issue(planner_state, action.issue_id)
        planner_state = _refresh_issues(
            planner_state,
            latest=_build_issues(bundle=bundle, inspect_state=inspect_state),
        )
        actions_run += 1

    return inspect_state, planner_state, str(trace_path)


def run_agent_review_pass(
    *,
    inspect_state: Mapping[str, object],
    tooling_runtime: object,
    bundle: MultitrackDescriptionBundle,
    max_actions: int = 3,
) -> tuple[PlannerState, str]:
    issues = _build_issues(bundle=bundle, inspect_state=inspect_state)
    planner_state = PlannerState(video_id=bundle.video_id, issues=issues)
    trace_path = _trace_path(bundle)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.touch(exist_ok=True)
    registry = {
        name: definition.handler
        for name, definition in build_tool_registry(tooling_runtime).items()
    }
    executor = ToolExecutor(registry=registry, trace_path=trace_path)

    actions_run = 0
    while actions_run < max_actions:
        action = plan_next_action(planner_state)
        if action is None:
            break
        planner_state = mark_issue_attempted(planner_state, action.issue_id)
        planner_state, result = executor.execute(planner_state, action)
        decision, confidence = _decision_for_result(action.tool_name, result)
        planner_state = executor.record_decision(
            planner_state,
            issue_id=action.issue_id,
            decision=decision,
            rationale=f"{action.tool_name} completed during bounded review pass",
            confidence=confidence,
        )
        if decision == "accept":
            planner_state = resolve_issue(planner_state, action.issue_id)
        actions_run += 1

    return planner_state, str(trace_path)


def _apply_action_result(
    *,
    inspect_state: InspectState,
    tooling_runtime: "ToolingRuntime",
    tool_name: str,
    result: object,
    registry: dict[str, Any],
) -> InspectState:
    updated = dict(inspect_state)
    if tool_name == "structural_overview" and isinstance(result, dict):
        updated["video_probe"] = result["probe"]
        updated["candidate_cuts"] = list(result["candidate_cuts"])
        updated["evidence_windows"] = list(result["evidence_windows"])
        updated["frame_batches"] = list(result["frame_batches"])
        updated["storyboard_path"] = str(result["storyboard_path"])
        return _rebuild_structural_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name in {"extract_entities", "recover_with_text_prompt"}:
        updated["sam3_track_set"] = _merge_track_sets(
            updated.get("sam3_track_set"),
            result,
        )
        return _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name == "crop_tracks":
        updated["track_crops"] = list(result)
        return _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name == "score_track_labels":
        updated["track_label_candidates"] = dict(result)
        return _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name == "validate_bundle":
        return updated

    return updated


def _rebuild_structural_state(
    inspect_state: InspectState,
    *,
    tooling_runtime: "ToolingRuntime",
    registry: dict[str, Any],
) -> InspectState:
    extraction = registry["extract_entities"](
        frame_batches=list(inspect_state.get("frame_batches", []))
    )
    inspect_state["sam3_track_set"] = extraction
    if tooling_runtime.runtime_profile == "mig10_safe":
        tooling_runtime.release_client("sam3")
    return _rebuild_semantic_state(inspect_state, tooling_runtime=tooling_runtime, registry=registry)


def _rebuild_semantic_state(
    inspect_state: InspectState,
    *,
    tooling_runtime: "ToolingRuntime",
    registry: dict[str, Any],
) -> InspectState:
    frame_batches = list(inspect_state.get("frame_batches", []))
    tracks = _tracks_from_result(inspect_state.get("sam3_track_set"))
    storyboard_path = str(inspect_state.get("storyboard_path", "storyboard.jpg"))
    track_crops = registry["crop_tracks"](
        frame_batches=frame_batches,
        tracks=tracks,
        output_dir=str(Path(storyboard_path).parent / "crops"),
    )
    inspect_state["track_crops"] = track_crops

    track_image_paths = group_crop_paths_by_track(track_crops)
    embeddings = (
        registry["embed_track_crops"](track_image_paths=track_image_paths)
        if track_image_paths
        else []
    )
    inspect_state["entity_embeddings"] = embeddings
    if tooling_runtime.runtime_profile == "mig10_safe":
        tooling_runtime.release_client("embedding")

    track_label_candidates = (
        registry["score_track_labels"](track_image_paths=track_image_paths)
        if track_image_paths
        else {}
    )
    inspect_state["track_label_candidates"] = track_label_candidates
    if tooling_runtime.runtime_profile == "mig10_safe":
        tooling_runtime.release_client("label")

    refined_structure = registry["refine_candidate_cuts"](
        probe=inspect_state["video_probe"],
        candidate_cuts=list(inspect_state.get("candidate_cuts", [])),
        frame_batches=frame_batches,
        tracks=tracks,
        label_candidates_by_track=track_label_candidates,
        storyboard_path=storyboard_path,
    )
    inspect_state["candidate_cuts"] = list(refined_structure["candidate_cuts"])
    inspect_state["evidence_windows"] = list(refined_structure["evidence_windows"])

    semantics = registry["build_source_semantics"](
        tracks=tracks,
        embeddings=embeddings,
        track_crops=track_crops,
        label_candidates_by_track=track_label_candidates,
        evidence_windows=list(inspect_state.get("evidence_windows", [])),
    )
    inspect_state["identity_edges"] = list(semantics["identity_edges"])
    inspect_state["physical_sources"] = list(semantics["physical_sources"])
    inspect_state["sound_event_segments"] = list(semantics["sound_events"])
    inspect_state["ambience_beds"] = list(semantics["ambience_beds"])
    inspect_state["generation_groups"] = list(semantics["generation_groups"])
    return inspect_state


def _refresh_issues(
    planner_state: PlannerState,
    *,
    latest: list[AgentIssue],
) -> PlannerState:
    by_id = {issue.issue_id: issue for issue in planner_state.issues}
    merged: list[AgentIssue] = []
    for issue in latest:
        existing = by_id.get(issue.issue_id)
        if existing is not None:
            issue.attempts = existing.attempts
            issue.status = existing.status
        merged.append(issue)
    planner_state.issues = merged
    return planner_state


def _merge_track_sets(existing: object, new_result: object) -> object:
    existing_tracks = _tracks_from_result(existing)
    new_tracks = _tracks_from_result(new_result)
    merged_tracks = {track.track_id: track for track in existing_tracks}
    for track in new_tracks:
        merged_tracks[track.track_id] = track
    if hasattr(new_result, "model_copy"):
        return new_result.model_copy(update={"tracks": list(merged_tracks.values())})
    if isinstance(new_result, dict):
        return {**new_result, "tracks": list(merged_tracks.values())}
    return new_result


def _tracks_from_result(result: object) -> list[object]:
    if isinstance(result, dict):
        tracks = result.get("tracks", [])
    else:
        tracks = getattr(result, "tracks", [])
    return list(tracks or [])


def _build_issues(
    *,
    bundle: MultitrackDescriptionBundle,
    inspect_state: Mapping[str, object],
) -> list[AgentIssue]:
    issues: list[AgentIssue] = []
    if not bundle.evidence_windows:
        issues.append(
            AgentIssue(
                issue_id="structural-gap",
                issue_type="structural_gap",
                description="No evidence windows available",
                priority=0,
                payload={"video_path": inspect_state.get("video_path", "")},
            )
        )
    if inspect_state.get("sam3_track_set") and not inspect_state.get("track_crops"):
        issues.append(
            AgentIssue(
                issue_id="missing-crops",
                issue_type="missing_crops",
                description="Tracks exist but crop artifacts are missing",
                priority=10,
                payload={
                    "frame_batches": list(inspect_state.get("frame_batches", [])),
                    "tracks": list(
                        getattr(inspect_state.get("sam3_track_set"), "tracks", [])
                    ),
                    "output_dir": str(_trace_path(bundle).parent / "crops"),
                },
            )
        )
    if inspect_state.get("track_crops") and not inspect_state.get("track_label_candidates"):
        issues.append(
            AgentIssue(
                issue_id="missing-labels",
                issue_type="missing_labels",
                description="Crop evidence exists but labels are missing",
                priority=15,
                payload={
                    "track_image_paths": group_crop_paths_by_track(
                        list(inspect_state.get("track_crops", []))
                    )
                },
            )
        )
    for index, issue in enumerate(bundle.validation.issues):
        issue_id = issue.issue_id or f"validation-{index:04d}"
        if issue.issue_type == "route_inconsistency":
            payload = {
                "tracks": list(getattr(inspect_state.get("sam3_track_set"), "tracks", []))
            }
            issue_type = "routing_ambiguity"
        elif issue.issue_type == "low_confidence_identity_merge":
            payload = {
                "frame_batches": list(inspect_state.get("frame_batches", [])),
                "text_prompt": _recovery_prompt(
                    bundle.physical_sources, issue.related_ids
                ),
            }
            issue_type = "ambiguous_source"
        else:
            payload = {"bundle": bundle}
            issue_type = "validation_issue"
        issues.append(
            AgentIssue(
                issue_id=issue_id,
                issue_type=issue_type,
                description=issue.message,
                priority=20 + index,
                payload=payload,
            )
        )
    return issues


def _decision_for_result(tool_name: str, result: object) -> tuple[str, float]:
    if tool_name == "validate_bundle":
        issue_count = len(result) if isinstance(result, list) else 1
        return (
            "accept" if issue_count == 0 else "retry",
            0.6 if issue_count == 0 else 0.4,
        )
    return ("accept", 0.8)


def _recovery_prompt(
    physical_sources: list[PhysicalSourceTrack],
    related_ids: list[str],
) -> str:
    for source in physical_sources:
        if source.source_id in related_ids and source.label_candidates:
            return source.label_candidates[0].label
    return "object"


def _trace_path(bundle: MultitrackDescriptionBundle) -> Path:
    root = Path(bundle.artifacts.storyboard_dir or ".")
    return root / f"{bundle.video_id}-agent-trace.jsonl"
