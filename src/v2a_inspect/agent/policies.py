from __future__ import annotations

from v2a_inspect.agent.state import AgentIssue, PlannerState


def can_retry_issue(state: PlannerState, issue: AgentIssue) -> bool:
    limit = _limit_for_issue(state, issue)
    return issue.attempts < limit


def _limit_for_issue(state: PlannerState, issue: AgentIssue) -> int:
    if issue.issue_type == "ambiguous_source":
        return state.retry_budget.manual_recovery_limit
    if issue.issue_type == "grouping_ambiguity":
        return state.retry_budget.regroup_retry_limit
    if issue.issue_type == "validation_issue":
        return state.retry_budget.validation_round_limit
    return state.retry_budget.extraction_retry_limit
