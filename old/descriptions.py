from __future__ import annotations

from v2a_inspect.contracts import (
    AmbienceBed,
    GenerationGroup,
    PhysicalSourceTrack,
    SoundEventSegment,
)

from .description_writer import DescriptionWriterLike


def synthesize_canonical_descriptions(
    generation_groups: list[GenerationGroup],
    *,
    sound_events: list[SoundEventSegment],
    ambience_beds: list[AmbienceBed],
    physical_sources: list[PhysicalSourceTrack],
    description_writer: DescriptionWriterLike | None = None,
    preserve_existing_descriptions: bool = False,
    stats: dict[str, int] | None = None,
) -> list[GenerationGroup]:
    del sound_events, ambience_beds, physical_sources
    if stats is not None:
        stats.setdefault("description_writer_calls", 0)
        stats.setdefault("preserved_description_count", 0)
        stats.setdefault("description_writer_failures", 0)
    if preserve_existing_descriptions:
        return [
            group
            for group in generation_groups
            if group.canonical_description
            and not group.description_stale
            and group.description_origin in {"writer", "manual"}
        ]
    if description_writer is None:
        return generation_groups
    updated: list[GenerationGroup] = []
    for group in generation_groups:
        if stats is not None:
            stats["description_writer_calls"] += 1
        draft = description_writer.write_group_description(
            {
                "group_id": group.group_id,
                "canonical_label": group.canonical_label,
                "member_event_ids": list(group.member_event_ids),
                "member_ambience_ids": list(group.member_ambience_ids),
                "route_prior": None
                if group.route_decision is None
                else group.route_decision.model_type,
            }
        )
        if draft is None:
            if stats is not None:
                stats["description_writer_failures"] += 1
            updated.append(group)
            continue
        updated.append(
            group.model_copy(
                update={
                    "canonical_description": draft.canonical_description,
                    "description_origin": "writer",
                    "description_stale": False,
                    "description_confidence": round(draft.description_confidence, 4),
                    "description_rationale": draft.description_rationale,
                }
            )
        )
    return updated
