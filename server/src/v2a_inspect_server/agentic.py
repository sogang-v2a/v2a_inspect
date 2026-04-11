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
from v2a_inspect.agent.policies import can_retry_issue
from v2a_inspect.contracts import MultitrackDescriptionBundle, PhysicalSourceTrack
from v2a_inspect.workflows import InspectState

from .crops import group_crop_paths_by_track
from .finalize import build_final_bundle
from .telemetry import record_recovery_attempt, record_stage, stage_start
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
    bundle = inspect_state.get("multitrack_bundle") or build_final_bundle(
        inspect_state,
        description_writer=getattr(tooling_runtime, "description_writer", None),
    )
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
        issue = next(
            (candidate for candidate in planner_state.issues if candidate.issue_id == action.issue_id),
            None,
        )
        adjudication = _adjudicate_issue(
            inspect_state=inspect_state,
            bundle=bundle,
            issue=issue,
            planned_tool_name=action.tool_name,
            tooling_runtime=tooling_runtime,
        )
        if adjudication is not None and adjudication.get("resolution") == "accept":
            inspect_state = _record_agent_review_decision(
                inspect_state,
                issue=issue,
                decision="accept",
                tool_name=action.tool_name,
                rationale=str(adjudication["rationale"]),
                confidence=float(adjudication["confidence"]),
            )
            if issue is not None and issue.issue_type in {"foreground_collapse", "missing_sources"}:
                inspect_state["terminal_resolution"] = "accepted_ambience_only"
            planner_state = mark_issue_attempted(planner_state, action.issue_id)
            planner_state = executor.record_decision(
                planner_state,
                issue_id=action.issue_id,
                decision="accept",
                rationale=str(adjudication["rationale"]),
                confidence=float(adjudication["confidence"]),
            )
            planner_state = resolve_issue(planner_state, action.issue_id)
            planner_state = _refresh_issues(
                planner_state,
                latest=_build_issues(bundle=bundle, inspect_state=inspect_state),
            )
            actions_run += 1
            continue
        tool_name = str(adjudication["tool_name"]) if adjudication is not None and adjudication.get("tool_name") else action.tool_name
        if tool_name != action.tool_name:
            action = action.model_copy(update={"tool_name": tool_name})
        planner_state = mark_issue_attempted(planner_state, action.issue_id)
        planner_state, result = executor.execute(planner_state, action)
        inspect_state = _apply_action_result(
            inspect_state=inspect_state,
            tooling_runtime=tooling_runtime,
            tool_name=action.tool_name,
            result=result,
            registry=registry,
        )
        bundle = build_final_bundle(
            inspect_state,
            description_writer=getattr(tooling_runtime, "description_writer", None),
        )
        inspect_state["multitrack_bundle"] = bundle

        current_issue_ids = {
            issue.issue_id
            for issue in _build_issues(bundle=bundle, inspect_state=inspect_state)
        }
        decision = "accept" if action.issue_id not in current_issue_ids else "retry"
        inspect_state = _record_agent_review_decision(
            inspect_state,
            issue=issue,
            decision=decision,
            tool_name=action.tool_name,
            rationale=f"{action.tool_name} completed during agentic tool loop",
            confidence=0.8 if decision == "accept" else 0.45,
        )
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

    inspect_state = _finalize_terminal_resolution(
        inspect_state=inspect_state,
        planner_state=planner_state,
    )
    bundle = build_final_bundle(
        inspect_state,
        description_writer=getattr(tooling_runtime, "description_writer", None),
    )
    inspect_state["multitrack_bundle"] = bundle
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
        updated["artifact_run_dir"] = str(result["artifact_root"])
        return _rebuild_structural_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name in {"extract_entities", "recover_with_text_prompt"}:
        started = stage_start()
        updated["sam3_track_set"] = _merge_track_sets(
            updated.get("sam3_track_set"),
            result,
        )
        updated.setdefault("recovery_actions", []).append(tool_name)
        if tool_name == "recover_with_text_prompt":
            record_recovery_attempt(
                updated,
                tool_name=tool_name,
                details={"track_count": len(_tracks_from_result(updated.get("sam3_track_set")))},
            )
        rebuilt = _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)
        record_stage(
            rebuilt,
            stage=f"agent:{tool_name}",
            started_at=started,
            metrics={"track_count": len(_tracks_from_result(rebuilt.get("sam3_track_set")))},
        )
        return rebuilt

    if tool_name == "recover_foreground_sources" and isinstance(result, dict):
        started = stage_start()
        updated["sam3_track_set"] = _merge_track_sets(
            updated.get("sam3_track_set"),
            result.get("track_set"),
        )
        updated["scene_prompt_candidates"] = dict(result.get("prompts_by_scene", {}))
        updated.setdefault("recovery_actions", []).append(tool_name)
        record_recovery_attempt(
            updated,
            tool_name=tool_name,
            details={
                "window_count": len(updated.get("evidence_windows", [])),
                "prompts_by_scene": dict(result.get("prompts_by_scene", {})),
                "track_count": len(_tracks_from_result(updated.get("sam3_track_set"))),
            },
        )
        rebuilt = _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)
        record_stage(
            rebuilt,
            stage="agent:recover_foreground_sources",
            started_at=started,
            metrics={"track_count": len(_tracks_from_result(rebuilt.get("sam3_track_set")))},
        )
        return rebuilt

    if tool_name == "crop_tracks":
        updated["track_crops"] = list(result)
        return _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name == "score_track_labels":
        updated["track_label_candidates"] = dict(result)
        return _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name == "group_embeddings":
        updated["candidate_groups"] = list(getattr(result, "groups", []))
        return _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name == "routing_priors":
        updated["track_routing_decisions"] = dict(result)
        return _rebuild_semantic_state(updated, tooling_runtime=tooling_runtime, registry=registry)

    if tool_name == "refine_candidate_cuts" and isinstance(result, dict):
        updated["candidate_cuts"] = list(result["candidate_cuts"])
        updated["evidence_windows"] = list(result["evidence_windows"])
        semantics = registry["build_source_semantics"](
            tracks=_tracks_from_result(updated.get("sam3_track_set")),
            embeddings=list(updated.get("entity_embeddings", [])),
            track_crops=list(updated.get("track_crops", [])),
            label_candidates_by_track=dict(updated.get("track_label_candidates", {})),
            evidence_windows=list(updated.get("evidence_windows", [])),
            candidate_groups=list(updated.get("candidate_groups", [])),
            routing_decisions_by_track=dict(updated.get("track_routing_decisions", {})),
        )
        updated["identity_edges"] = list(semantics["identity_edges"])
        updated["physical_sources"] = list(semantics["physical_sources"])
        updated["sound_event_segments"] = list(semantics["sound_events"])
        updated["ambience_beds"] = list(semantics["ambience_beds"])
        updated["generation_groups"] = list(semantics["generation_groups"])
        return updated

    if tool_name == "densify_window_sampling" and isinstance(result, dict):
        started = stage_start()
        updated["frame_batches"] = list(result["frame_batches"])
        updated["evidence_windows"] = list(result["evidence_windows"])
        updated["storyboard_path"] = str(result["storyboard_path"])
        updated["frames_per_window"] = int(result["frames_per_scene"])
        updated.setdefault("recovery_actions", []).append(tool_name)
        record_recovery_attempt(
            updated,
            tool_name=tool_name,
            details={
                "window_ids": list(result.get("window_ids", [])),
                "frames_per_scene": int(result["frames_per_scene"]),
            },
        )
        rebuilt = _rebuild_structural_state(updated, tooling_runtime=tooling_runtime, registry=registry)
        record_stage(
            rebuilt,
            stage="agent:densify_window_sampling",
            started_at=started,
            metrics={
                "window_count": len(rebuilt.get("evidence_windows", [])),
                "frames_per_window": rebuilt.get("frames_per_window"),
            },
        )
        return rebuilt

    if tool_name == "rerun_description_writer":
        updated["generation_groups"] = list(result)
        return updated

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
        frame_batches=list(inspect_state.get("frame_batches", [])),
        prompts_by_scene=dict(inspect_state.get("scene_prompt_candidates", {})) or None,
    )
    inspect_state["sam3_track_set"] = extraction
    if tooling_runtime.should_release_clients:
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
    candidate_group_result = (
        registry["group_embeddings"](
            embeddings=embeddings,
            tracks_by_id={track.track_id: track for track in tracks},
        )
        if embeddings
        else None
    )
    inspect_state["candidate_groups"] = list(
        getattr(candidate_group_result, "groups", [])
    )
    if tooling_runtime.should_release_clients:
        tooling_runtime.release_client("embedding")

    track_label_candidates = (
        registry["score_track_labels"](track_image_paths=track_image_paths)
        if track_image_paths
        else {}
    )
    inspect_state["track_label_candidates"] = track_label_candidates
    inspect_state["track_routing_decisions"] = (
        dict(registry["routing_priors"](tracks=tracks)) if tracks else {}
    )
    if tooling_runtime.should_release_clients:
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
        candidate_groups=list(inspect_state.get("candidate_groups", [])),
        routing_decisions_by_track=dict(inspect_state.get("track_routing_decisions", {})),
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
    current_tracks = list(_tracks_from_result(inspect_state.get("sam3_track_set")))
    current_sources = list(bundle.physical_sources)
    current_events = list(bundle.sound_events)
    duration_seconds = float(getattr(bundle.video_meta, "duration_seconds", 0.0))
    if duration_seconds > 1.0 and not current_tracks:
        issues.append(
            AgentIssue(
                issue_id="foreground-collapse",
                issue_type="foreground_collapse",
                description="No foreground tracks were extracted from a non-trivial clip.",
                priority=2,
                payload={
                    "video_path": inspect_state.get("video_path", ""),
                    "evidence_windows": list(bundle.evidence_windows),
                    "frame_batches": list(inspect_state.get("frame_batches", [])),
                    "window_ids": [window.window_id for window in bundle.evidence_windows],
                    "output_root": bundle.artifacts.run_dir,
                    "storyboard_path": bundle.artifacts.storyboard_path,
                    "current_track_count": 0,
                    "current_source_count": len(current_sources),
                    "current_event_count": len(current_events),
                    "frames_per_scene": int(inspect_state.get("frames_per_window", 3)),
                    "recovery_actions": list(inspect_state.get("recovery_actions", [])),
                    "recovery_attempts": list(inspect_state.get("recovery_attempts", [])),
                    "scene_prompt_candidates": dict(inspect_state.get("scene_prompt_candidates", {})),
                    "scene_prompt_recovery_attempted": _has_recovery_attempt(
                        inspect_state, "recover_foreground_sources"
                    ),
                    "text_recovery_attempted": _has_recovery_attempt(
                        inspect_state, "recover_with_text_prompt"
                    ),
                    "text_prompt": _foreground_recovery_prompt(inspect_state),
                },
            )
        )
    elif duration_seconds > 1.0 and current_tracks and not current_sources:
        issues.append(
            AgentIssue(
                issue_id="missing-sources",
                issue_type="missing_sources",
                description="Tracks exist but no physical sources were built from them.",
                priority=4,
                payload={
                    "frame_batches": list(inspect_state.get("frame_batches", [])),
                    "current_track_count": len(current_tracks),
                    "current_source_count": 0,
                    "current_event_count": len(current_events),
                    "scene_prompt_candidates": dict(inspect_state.get("scene_prompt_candidates", {})),
                    "recovery_actions": list(inspect_state.get("recovery_actions", [])),
                    "recovery_attempts": list(inspect_state.get("recovery_attempts", [])),
                    "scene_prompt_recovery_attempted": _has_recovery_attempt(
                        inspect_state, "recover_foreground_sources"
                    ),
                    "text_recovery_attempted": _has_recovery_attempt(
                        inspect_state, "recover_with_text_prompt"
                    ),
                    "text_prompt": _foreground_recovery_prompt(inspect_state),
                },
            )
        )
    if current_tracks and not inspect_state.get("track_crops"):
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
    low_confidence_cuts = [
        cut
        for cut in bundle.candidate_cuts
        if cut.confidence < 0.6
    ]
    overly_broad_windows = [
        window
        for window in bundle.evidence_windows
        if (window.end_time - window.start_time) > 8.0
    ]
    if low_confidence_cuts or overly_broad_windows:
        issues.append(
            AgentIssue(
                issue_id="cut-ambiguity",
                issue_type="cut_ambiguity",
                description="Structural windows have low-confidence or over-broad cut boundaries",
                priority=12,
                payload={
                    "probe": inspect_state.get("video_probe"),
                    "candidate_cuts": list(inspect_state.get("candidate_cuts", [])),
                    "frame_batches": list(inspect_state.get("frame_batches", [])),
                    "tracks": list(getattr(inspect_state.get("sam3_track_set"), "tracks", [])),
                    "label_candidates_by_track": dict(inspect_state.get("track_label_candidates", {})),
                    "storyboard_path": inspect_state.get("storyboard_path"),
                    "low_confidence_cut_ids": [cut.cut_id for cut in low_confidence_cuts],
                    "broad_window_ids": [window.window_id for window in overly_broad_windows],
                },
            )
        )
    for index, issue in enumerate(bundle.validation.issues):
        issue_id = issue.issue_id or f"validation-{index:04d}"
        if issue.issue_type == "missing_dominant_source":
            continue
        if issue.issue_type in {"recovery_exhausted", "accepted_ambience_only"}:
            continue
        if issue.issue_type == "route_inconsistency":
            payload = {
                "tracks": list(getattr(inspect_state.get("sam3_track_set"), "tracks", []))
            }
            issue_type = "routing_ambiguity"
        elif issue.issue_type == "suspicious_cross_scene_generation_merge":
            payload = {
                "embeddings": list(inspect_state.get("entity_embeddings", [])),
                "tracks_by_id": {
                    track.track_id: track
                    for track in list(
                        getattr(inspect_state.get("sam3_track_set"), "tracks", [])
                    )
                },
            }
            issue_type = "grouping_ambiguity"
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
    stale_group_ids = [
        group.group_id
        for group in bundle.generation_groups
        if getattr(group, "description_stale", False)
    ]
    if stale_group_ids:
        issues.append(
            AgentIssue(
                issue_id="description-stale",
                issue_type="description_stale",
                description="Some generation groups need a fresh description rewrite",
                priority=18,
                payload={
                    "generation_groups": list(bundle.generation_groups),
                    "sound_events": list(bundle.sound_events),
                    "ambience_beds": list(bundle.ambience_beds),
                    "physical_sources": list(bundle.physical_sources),
                    "stale_group_ids": stale_group_ids,
                },
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
    if bundle.artifacts.trace_path:
        return Path(bundle.artifacts.trace_path)
    root = Path(bundle.artifacts.run_dir or ".")
    return root / f"{bundle.video_id}-agent-trace.jsonl"


def _adjudicate_issue(
    *,
    inspect_state: Mapping[str, object],
    bundle: MultitrackDescriptionBundle,
    issue: AgentIssue | None,
    planned_tool_name: str,
    tooling_runtime: "ToolingRuntime",
) -> dict[str, object] | None:
    if issue is None or issue.issue_type not in {
        "cut_ambiguity",
        "foreground_collapse",
        "missing_sources",
        "grouping_ambiguity",
        "routing_ambiguity",
        "description_stale",
    }:
        return None
    judge = getattr(tooling_runtime, "adjudication_judge", None)
    if judge is None:
        return None
    decision = judge.judge_issue(
        {
            "video_id": bundle.video_id,
            "issue_id": issue.issue_id,
            "issue_type": issue.issue_type,
            "issue_attempts": issue.attempts,
            "issue_description": issue.description,
            "planned_tool_name": planned_tool_name,
            "current_frames_per_window": inspect_state.get("frames_per_window"),
            "current_track_count": len(_tracks_from_result(inspect_state.get("sam3_track_set"))),
            "current_source_count": len(bundle.physical_sources),
            "current_event_count": len(bundle.sound_events),
            "scene_prompt_candidates_present": bool(inspect_state.get("scene_prompt_candidates")),
            "recovery_actions": list(inspect_state.get("recovery_actions", []))[-3:],
            "recovery_attempts": list(inspect_state.get("recovery_attempts", []))[-3:],
            "stage_history": list(inspect_state.get("stage_history", []))[-5:],
            "validation_issues": [item.model_dump(mode="json") for item in bundle.validation.issues],
            "candidate_cut_count": len(bundle.candidate_cuts),
            "evidence_window_count": len(bundle.evidence_windows),
            "generation_group_count": len(bundle.generation_groups),
            "issue_payload": issue.payload,
        }
    )
    if decision is None:
        return None
    return decision.model_dump(mode="json")


def _has_recovery_attempt(
    inspect_state: Mapping[str, object],
    tool_name: str,
) -> bool:
    return any(
        isinstance(attempt, Mapping) and attempt.get("tool_name") == tool_name
        for attempt in list(inspect_state.get("recovery_attempts", []))
    )


def _foreground_recovery_prompt(inspect_state: Mapping[str, object]) -> str:
    prompts_by_scene = inspect_state.get("scene_prompt_candidates", {})
    if isinstance(prompts_by_scene, Mapping):
        for prompts in prompts_by_scene.values():
            if isinstance(prompts, list):
                for prompt in prompts:
                    if isinstance(prompt, str) and prompt.strip():
                        return prompt.strip()
    return "object"


def _record_agent_review_decision(
    inspect_state: InspectState,
    *,
    issue: AgentIssue | None,
    decision: str,
    tool_name: str,
    rationale: str,
    confidence: float,
) -> InspectState:
    updated = dict(inspect_state)
    decisions = list(updated.get("agent_review_decisions", []))
    decisions.append(
        {
            "issue_id": issue.issue_id if issue is not None else None,
            "issue_type": issue.issue_type if issue is not None else None,
            "decision": decision,
            "tool_name": tool_name,
            "confidence": confidence,
            "rationale": rationale,
        }
    )
    updated["agent_review_decisions"] = decisions
    if decision != "accept" and updated.get("terminal_resolution") == "accepted_ambience_only":
        updated.pop("terminal_resolution", None)
    return updated


def _finalize_terminal_resolution(
    *,
    inspect_state: InspectState,
    planner_state: PlannerState,
) -> InspectState:
    updated = dict(inspect_state)
    if updated.get("terminal_resolution") == "accepted_ambience_only":
        return updated
    open_foreground_issues = [
        issue
        for issue in planner_state.issues
        if issue.status == "open" and issue.issue_type in {"foreground_collapse", "missing_sources"}
    ]
    if any(not can_retry_issue(planner_state, issue) for issue in open_foreground_issues):
        updated["terminal_resolution"] = "recovery_exhausted"
        return updated
    if "terminal_resolution" in updated and updated["terminal_resolution"] == "recovery_exhausted":
        updated.pop("terminal_resolution", None)
    return updated
