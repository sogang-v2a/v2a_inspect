from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from v2a_inspect.contracts import (
    LabelCandidate,
    MultitrackDescriptionBundle,
    ReviewEdit,
    RoutingDecision,
    ValidationIssue,
    ValidationReport,
)


def persist_bundle(bundle: MultitrackDescriptionBundle, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def load_bundle(path: str | Path) -> MultitrackDescriptionBundle:
    return MultitrackDescriptionBundle.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def apply_route_override(
    bundle: MultitrackDescriptionBundle,
    *,
    group_id: str,
    model_type: str,
    author: str | None = None,
    rationale: str = "manual route override",
) -> MultitrackDescriptionBundle:
    updated = bundle.model_copy(deep=True)
    for group in updated.generation_groups:
        if group.group_id != group_id:
            continue
        group.route_decision = RoutingDecision(
            model_type=model_type,
            confidence=1.0,
            factors=["manual_override"],
            reasoning=rationale,
            rule_based=False,
        )
        _append_edit(
            updated,
            action="route_override",
            target_ids=[group_id],
            payload={"model_type": model_type},
            author=author,
            rationale=rationale,
        )
        break
    return _normalize_bundle(updated)


def split_generation_group(
    bundle: MultitrackDescriptionBundle,
    *,
    group_id: str,
    event_ids: list[str],
    author: str | None = None,
    rationale: str = "manual split",
) -> MultitrackDescriptionBundle:
    updated = bundle.model_copy(deep=True)
    for index, group in enumerate(updated.generation_groups):
        if group.group_id != group_id:
            continue
        moved_events = [
            event_id for event_id in group.member_event_ids if event_id in event_ids
        ]
        if not moved_events or len(moved_events) == len(group.member_event_ids):
            return updated
        group.member_event_ids = [
            event_id
            for event_id in group.member_event_ids
            if event_id not in moved_events
        ]
        new_group = group.model_copy(
            update={
                "group_id": f"{group_id}-split-{len(updated.review_metadata.applied_edits):02d}",
                "member_event_ids": moved_events,
                "canonical_label": f"{group.canonical_label}:split",
                "canonical_description": f"{group.canonical_description} (split)",
            }
        )
        updated.generation_groups.insert(index + 1, new_group)
        _append_edit(
            updated,
            action="split_generation_group",
            target_ids=[group_id, new_group.group_id],
            payload={"event_ids": moved_events},
            author=author,
            rationale=rationale,
        )
        break
    return _normalize_bundle(updated)


def merge_generation_groups(
    bundle: MultitrackDescriptionBundle,
    *,
    source_group_ids: list[str],
    author: str | None = None,
    rationale: str = "manual merge",
) -> MultitrackDescriptionBundle:
    updated = bundle.model_copy(deep=True)
    groups = [
        group
        for group in updated.generation_groups
        if group.group_id in source_group_ids
    ]
    if len(groups) < 2:
        return updated
    primary = groups[0]
    for secondary in groups[1:]:
        primary.member_event_ids = sorted(
            set(primary.member_event_ids + secondary.member_event_ids)
        )
        primary.member_ambience_ids = sorted(
            set(primary.member_ambience_ids + secondary.member_ambience_ids)
        )
    updated.generation_groups = [
        group
        for group in updated.generation_groups
        if group.group_id == primary.group_id or group.group_id not in source_group_ids
    ]
    _append_edit(
        updated,
        action="merge_generation_groups",
        target_ids=source_group_ids,
        payload={"primary_group_id": primary.group_id},
        author=author,
        rationale=rationale,
    )
    return _normalize_bundle(updated)


def rename_source(
    bundle: MultitrackDescriptionBundle,
    *,
    source_id: str,
    new_label: str,
    author: str | None = None,
    rationale: str = "manual rename",
) -> MultitrackDescriptionBundle:
    updated = bundle.model_copy(deep=True)
    for source in updated.physical_sources:
        if source.source_id != source_id:
            continue
        if source.label_candidates:
            source.label_candidates[0] = LabelCandidate(label=new_label, score=1.0)
        else:
            source.label_candidates.append(LabelCandidate(label=new_label, score=1.0))
        _append_edit(
            updated,
            action="rename_source",
            target_ids=[source_id],
            payload={"new_label": new_label},
            author=author,
            rationale=rationale,
        )
        break
    return _normalize_bundle(updated)


def approve_validation_issue(
    bundle: MultitrackDescriptionBundle,
    *,
    issue_id: str,
    author: str | None = None,
    rationale: str = "manual approval",
) -> MultitrackDescriptionBundle:
    updated = bundle.model_copy(deep=True)
    if issue_id not in updated.validation.reviewed_issue_ids:
        updated.validation.reviewed_issue_ids.append(issue_id)
    _append_edit(
        updated,
        action="approve_issue",
        target_ids=[issue_id],
        payload={},
        author=author,
        rationale=rationale,
    )
    return _normalize_bundle(updated, preserve_reviewed_issue_ids=True)


def _append_edit(
    bundle: MultitrackDescriptionBundle,
    *,
    action: str,
    target_ids: list[str],
    payload: dict[str, object],
    author: str | None,
    rationale: str,
) -> None:
    bundle.review_metadata.applied_edits.append(
        ReviewEdit(
            edit_id=f"edit-{len(bundle.review_metadata.applied_edits):04d}",
            action=action,
            target_ids=target_ids,
            payload=payload,
            author=author,
            rationale=rationale,
            created_at=datetime.now(UTC).isoformat(),
        )
    )


def _normalize_bundle(
    bundle: MultitrackDescriptionBundle,
    *,
    preserve_reviewed_issue_ids: bool = False,
) -> MultitrackDescriptionBundle:
    updated = bundle.model_copy(deep=True)
    sources_by_id = {source.source_id: source for source in updated.physical_sources}
    events_by_id = {event.event_id: event for event in updated.sound_events}
    ambience_by_id = {ambience.ambience_id: ambience for ambience in updated.ambience_beds}

    for group in updated.generation_groups:
        if group.member_event_ids:
            member_events = [
                events_by_id[event_id]
                for event_id in group.member_event_ids
                if event_id in events_by_id
            ]
            source_labels = [
                sources_by_id[event.source_id].label_candidates[0].label
                for event in member_events
                if event.source_id in sources_by_id
                and sources_by_id[event.source_id].label_candidates
            ]
            event_types = [event.event_type for event in member_events]
            materials = [
                event.material_or_surface for event in member_events if event.material_or_surface
            ]
            patterns = [event.pattern for event in member_events if event.pattern]
            group.canonical_description = (
                f"{_majority_or_default(source_labels, default='source')} "
                f"{_majority_or_default(event_types, default='sound_event').replace('_', ' ')} "
                f"on {_majority_or_default(materials, default='generic')} "
                f"with {_majority_or_default(patterns, default='mixed')} texture"
            )
            group.description_confidence = round(group.group_confidence, 4)
            group.description_rationale = (
                "post-review normalization from source labels, event types, materials, and patterns"
            )
        elif group.member_ambience_ids:
            member_ambience = [
                ambience_by_id[ambience_id]
                for ambience_id in group.member_ambience_ids
                if ambience_id in ambience_by_id
            ]
            environment = _majority_or_default(
                [ambience.environment_type for ambience in member_ambience],
                default="environment",
            )
            texture = _majority_or_default(
                [ambience.acoustic_profile for ambience in member_ambience],
                default="continuous ambience",
            )
            group.canonical_description = f"{environment} ambience bed with {texture}"
            group.description_confidence = round(group.group_confidence, 4)
            group.description_rationale = (
                "post-review normalization from ambience environment and profile"
            )

        if "manual_override" not in group.route_decision.factors:
            group.route_decision = _normalized_route_decision(group)

    reviewed_issue_ids = (
        list(updated.validation.reviewed_issue_ids) if preserve_reviewed_issue_ids else []
    )
    issues = _validate_bundle(updated)
    updated.validation = ValidationReport(
        status=(
            "fail"
            if any(issue.severity == "error" for issue in issues)
            else ("pass_with_warnings" if issues else "pass")
        ),
        issues=issues,
        reviewed_issue_ids=reviewed_issue_ids,
    )
    return updated


def _normalized_route_decision(group: object) -> RoutingDecision:
    member_event_ids = list(getattr(group, "member_event_ids", []))
    member_ambience_ids = list(getattr(group, "member_ambience_ids", []))
    if member_ambience_ids:
        return RoutingDecision(
            model_type="TTA",
            confidence=0.9,
            factors=["ambience_bed", "post_review_normalization"],
            reasoning="background ambience defaults to TTA after review normalization",
            rule_based=True,
        )
    model_type = "VTA" if len(member_event_ids) <= 1 else "TTA"
    confidence = 0.75 if len(member_event_ids) <= 1 else 0.7
    factors = ["event_cardinality"]
    if len(member_event_ids) > 3:
        model_type = "TTA"
        confidence = max(confidence, 0.75)
        factors.append("crowded_group")
    elif len(member_event_ids) == 1:
        factors.append("single_dominant_event")
    return RoutingDecision(
        model_type=model_type,
        confidence=round(confidence, 4),
        factors=factors,
        reasoning="deterministic route normalization after review edits",
        rule_based=True,
    )


def _validate_bundle(bundle: MultitrackDescriptionBundle) -> list[ValidationIssue]:
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
        if (
            group.description_confidence is not None
            and group.description_confidence < 0.6
        ):
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
                sources=[sources_by_id[source_id] for source_id in sorted(source_ids) if source_id in sources_by_id]
            ):
                issues.append(
                    ValidationIssue(
                        issue_type="suspicious_cross_scene_generation_merge",
                        severity="warning",
                        message=f"{group.group_id} combines multiple contradictory physical sources",
                        related_ids=[group.group_id, *sorted(source_ids)],
                        recommended_action="split_group",
                        repair_tool="group_embeddings",
                    )
                )
        if (
            group.member_event_ids
            and len(group.member_event_ids) > 3
            and group.route_decision.model_type == "VTA"
        ):
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
    sources: list[object],
) -> bool:
    labels = {
        source.label_candidates[0].label
        for source in sources
        if getattr(source, "label_candidates", None)
    }
    if len(labels) > 1:
        return True
    spans = [span for source in sources for span in getattr(source, "spans", [])]
    for index, (left_start, left_end) in enumerate(spans):
        for right_start, right_end in spans[index + 1 :]:
            overlap = max(min(left_end, right_end) - max(left_start, right_start), 0.0)
            if overlap > 0.0:
                return True
    return False


def _majority_or_default(values: list[str], *, default: str) -> str:
    if not values:
        return default
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
