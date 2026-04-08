from __future__ import annotations

from pathlib import Path

from v2a_inspect.agent import (
    AgentIssue,
    PlannerState,
    ToolExecutor,
    mark_issue_attempted,
    plan_next_action,
    resolve_issue,
)
from v2a_inspect.contracts import MultitrackDescriptionBundle, PhysicalSourceTrack
from collections.abc import Mapping

from .tool_registry import build_tool_registry


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
                    "tracks": list(getattr(inspect_state.get("sam3_track_set"), "tracks", [])),
                    "output_dir": str(_trace_path(bundle).parent / "crops"),
                },
            )
        )
    for index, issue in enumerate(bundle.validation.issues):
        issue_id = issue.issue_id or f"validation-{index:04d}"
        if issue.issue_type == "route_inconsistency":
            payload = {"tracks": list(getattr(inspect_state.get("sam3_track_set"), "tracks", []))}
            issue_type = "routing_ambiguity"
        elif issue.issue_type == "low_confidence_identity_merge":
            payload = {
                "frame_batches": list(inspect_state.get("frame_batches", [])),
                "text_prompt": _recovery_prompt(bundle.physical_sources, issue.related_ids),
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
        return ("accept" if issue_count == 0 else "retry", 0.6 if issue_count == 0 else 0.4)
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
