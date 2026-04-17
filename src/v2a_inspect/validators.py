from __future__ import annotations

from v2a_inspect.contracts import MultitrackDescriptionBundle, ValidationIssue


def validate_bundle(bundle: MultitrackDescriptionBundle) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not bundle.generation_groups:
        issues.append(
            ValidationIssue(
                issue_id="empty-generation-groups",
                issue_type="empty_generation_groups",
                severity="warning",
                message="No generation groups were resolved from the current evidence.",
                recommended_action="rerun_tool",
            )
        )
        return issues
    for group in bundle.generation_groups:
        if group.route_decision is None:
            issues.append(
                ValidationIssue(
                    issue_id=f"unresolved-route-{group.group_id}",
                    issue_type="unresolved_route_decision",
                    severity="warning",
                    message=f"Generation group {group.group_id} has no routing decision yet.",
                    related_ids=[group.group_id],
                    recommended_action="rerun_tool",
                )
            )
        if not group.canonical_description:
            issues.append(
                ValidationIssue(
                    issue_id=f"unresolved-description-{group.group_id}",
                    issue_type="unresolved_description",
                    severity="warning",
                    message=f"Generation group {group.group_id} has no canonical description yet.",
                    related_ids=[group.group_id],
                    recommended_action="rerun_tool",
                )
            )
    return issues
