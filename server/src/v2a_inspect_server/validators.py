from __future__ import annotations

from v2a_inspect.contracts import MultitrackDescriptionBundle, ValidationIssue


def validate_bundle(bundle: MultitrackDescriptionBundle) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for source in bundle.physical_sources:
        if source.identity_confidence < 0.6:
            issues.append(
                ValidationIssue(
                    issue_type="low_confidence_identity_merge",
                    severity="warning",
                    message=f"{source.source_id} has low identity confidence",
                    related_ids=[source.source_id],
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
                )
            )
    return issues
