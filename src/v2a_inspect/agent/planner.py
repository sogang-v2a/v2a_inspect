from __future__ import annotations

from v2a_inspect.agent.policies import can_retry_issue
from v2a_inspect.agent.state import PlannedAction, PlannerState


_TOOL_BY_ISSUE = {
    "structural_gap": "structural_overview",
    "cut_ambiguity": "refine_candidate_cuts",
    "ambiguous_source": "recover_with_text_prompt",
    "missing_crops": "crop_tracks",
    "missing_labels": "score_track_labels",
    "grouping_ambiguity": "group_embeddings",
    "routing_ambiguity": "routing_priors",
    "description_stale": "rerun_description_writer",
    "validation_issue": "validate_bundle",
}


def plan_next_action(state: PlannerState) -> PlannedAction | None:
    open_issues = sorted(
        [issue for issue in state.issues if issue.status == "open"],
        key=lambda issue: issue.priority,
    )
    for issue in open_issues:
        if not can_retry_issue(state, issue):
            continue
        tool_name = _TOOL_BY_ISSUE[issue.issue_type]
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
