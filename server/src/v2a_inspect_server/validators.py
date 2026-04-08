from __future__ import annotations

from v2a_inspect.contracts import MultitrackDescriptionBundle, ValidationIssue


def validate_bundle(bundle: MultitrackDescriptionBundle) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not bundle.physical_sources:
        issues.append(
            ValidationIssue(
                issue_type="missing_dominant_source",
                severity="warning",
                message="Bundle has no physical sources.",
                recommended_action="rerun_tool",
                repair_tool="extract_entities",
            )
        )
    for source in bundle.physical_sources:
        if source.identity_confidence < 0.6:
            issues.append(
                ValidationIssue(
                    issue_type="low_confidence_identity_merge",
                    severity="warning",
                    message=f"{source.source_id} has low identity confidence",
                    related_ids=[source.source_id],
                    recommended_action="human_review",
                    repair_tool="build_source_semantics",
                )
            )
    for group in bundle.generation_groups:
        if not group.member_event_ids and not group.member_ambience_ids:
            issues.append(
                ValidationIssue(
                    issue_type="overlapping_contradictory_assignments",
                    severity="error",
                    message=f"{group.group_id} has no members",
                    related_ids=[group.group_id],
                    recommended_action="split_group",
                    repair_tool="build_source_semantics",
                )
            )
        if group.description_confidence is not None and group.description_confidence < 0.6:
            issues.append(
                ValidationIssue(
                    issue_type="overly_vague_description",
                    severity="warning",
                    message=f"{group.group_id} has a low-confidence canonical description",
                    related_ids=[group.group_id],
                    recommended_action="human_review",
                )
            )
        if group.member_event_ids and len(group.member_event_ids) > 3 and group.route_decision.model_type == "VTA":
            issues.append(
                ValidationIssue(
                    issue_type="route_inconsistency",
                    severity="warning",
                    message=f"{group.group_id} is crowded but routed to VTA",
                    related_ids=[group.group_id],
                    recommended_action="override_route",
                )
            )
    return issues
