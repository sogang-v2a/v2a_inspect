from __future__ import annotations

from v2a_inspect.agent.policies import can_retry_issue
from v2a_inspect.agent.state import AgentIssue, PlannedAction, PlannerState


_TOOL_BY_ISSUE = {
    "structural_gap": "structural_overview",
    "ambiguous_source": "propose_source_hypotheses",
    "hypothesis_conflict": "propose_source_hypotheses",
    "missing_sources": "build_source_semantics",
    "grouping_ambiguity": "build_source_semantics",
    "routing_ambiguity": "build_source_semantics",
    "description_stale": "rerun_description_writer",
}


def plan_next_action(state: PlannerState) -> PlannedAction | None:
    open_issues = sorted(
        [issue for issue in state.issues if issue.status == "open"],
        key=lambda issue: issue.priority,
    )
    for issue in open_issues:
        if not can_retry_issue(state, issue):
            continue
        tool_name = _tool_for_issue(issue)
        return PlannedAction(
            issue_id=issue.issue_id,
            tool_name=tool_name,
            request_payload=dict(issue.payload),
            rationale=f"Address {issue.issue_type} via {tool_name}",
        )
    return None


def mark_issue_attempted(state: PlannerState, issue_id: str) -> PlannerState:
    updated = state.model_copy(deep=True)
    for issue in updated.issues:
        if issue.issue_id == issue_id:
            issue.attempts += 1
            break
    return updated


def resolve_issue(state: PlannerState, issue_id: str) -> PlannerState:
    updated = state.model_copy(deep=True)
    for issue in updated.issues:
        if issue.issue_id == issue_id:
            issue.status = "resolved"
            break
    return updated


def _tool_for_issue(issue: AgentIssue) -> str:
    if issue.issue_type == "foreground_collapse":
        if issue.attempts <= 0 and _payload_int(issue.payload, "frames_per_scene", 2) < 6:
            return "densify_window_sampling"
        if _payload_int(issue.payload, "region_seed_count", 0) > 0:
            return "extract_entities"
        return "propose_source_hypotheses"
    return _TOOL_BY_ISSUE[issue.issue_type]


def _payload_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    return value if isinstance(value, int) else default
