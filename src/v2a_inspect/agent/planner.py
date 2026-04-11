from __future__ import annotations

from v2a_inspect.agent.policies import can_retry_issue
from v2a_inspect.agent.state import AgentIssue, PlannedAction, PlannerState


_TOOL_BY_ISSUE = {
    "structural_gap": "structural_overview",
    "cut_ambiguity": "refine_candidate_cuts",
    "ambiguous_source": "recover_with_text_prompt",
    "missing_sources": "recover_foreground_sources",
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
        if issue.attempts <= 0 and int(issue.payload.get("frames_per_scene", 2)) < 4:
            return "densify_window_sampling"
        if not bool(issue.payload.get("scene_prompt_recovery_attempted")):
            return "recover_foreground_sources"
        return "recover_with_text_prompt"
    if issue.issue_type == "missing_sources":
        if not bool(issue.payload.get("scene_prompt_recovery_attempted")):
            return "recover_foreground_sources"
        return "recover_with_text_prompt"
    return _TOOL_BY_ISSUE[issue.issue_type]
