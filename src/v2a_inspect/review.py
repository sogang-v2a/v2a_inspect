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
                "canonical_description": group.canonical_description,
                "description_stale": True,
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
        _mark_groups_description_stale(
            updated,
            target_group_ids=[group_id, new_group.group_id],
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
    _mark_groups_description_stale(updated, target_group_ids=[primary.group_id])
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
        related_event_ids = {
            event.event_id
            for event in updated.sound_events
            if event.source_id == source_id
        }
        _mark_groups_description_stale(
            updated,
            target_group_ids=[
                group.group_id
                for group in updated.generation_groups
                if any(event_id in related_event_ids for event_id in group.member_event_ids)
            ],
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
    del preserve_reviewed_issue_ids
    return bundle.model_copy(deep=True)


def _mark_groups_description_stale(
    bundle: MultitrackDescriptionBundle,
    *,
    target_group_ids: list[str],
) -> None:
    target_ids = set(target_group_ids)
    for group in bundle.generation_groups:
        if group.group_id in target_ids and group.description_origin == "writer":
            group.description_stale = True


def _normalized_route_decision(group: object) -> RoutingDecision:
    existing = getattr(group, "route_decision", None)
    return existing if isinstance(existing, RoutingDecision) else RoutingDecision.model_validate(existing)


def _validate_bundle(bundle: MultitrackDescriptionBundle) -> list[ValidationIssue]:
    del bundle
    return []


def _majority_or_default(values: list[str], *, default: str) -> str:
    del values, default
    return ""
