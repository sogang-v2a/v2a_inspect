from __future__ import annotations

from v2a_inspect.contracts import MultitrackDescriptionBundle, ValidationIssue


def validate_bundle(bundle: MultitrackDescriptionBundle) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    events_by_id = {event.event_id: event for event in bundle.sound_events}
    sources_by_id = {source.source_id: source for source in bundle.physical_sources}
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
        if group.member_event_ids:
            source_ids = {
                events_by_id[event_id].source_id
                for event_id in group.member_event_ids
                if event_id in events_by_id
            }
            if len(source_ids) > 1 and _is_suspicious_generation_merge(
                source_ids=source_ids,
                sources_by_id=sources_by_id,
            ):
                issues.append(
                    ValidationIssue(
                        issue_type="suspicious_cross_scene_generation_merge",
                        severity="warning",
                        message=f"{group.group_id} combines multiple physical sources",
                        related_ids=[group.group_id, *sorted(source_ids)],
                        recommended_action="split_group",
                        repair_tool="group_embeddings",
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


def _is_suspicious_generation_merge(
    *,
    source_ids: set[str],
    sources_by_id: dict[str, object],
) -> bool:
    sources = [sources_by_id[source_id] for source_id in sorted(source_ids) if source_id in sources_by_id]
    if len(sources) < 2:
        return False
    labels = {
        source.label_candidates[0].label
        for source in sources
        if getattr(source, "label_candidates", None)
    }
    if len(labels) > 1:
        return True
    spans = [
        span
        for source in sources
        for span in getattr(source, "spans", [])
    ]
    for index, (left_start, left_end) in enumerate(spans):
        for right_start, right_end in spans[index + 1 :]:
            overlap = max(min(left_end, right_end) - max(left_start, right_start), 0.0)
            if overlap > 0.0:
                return True
    return False
