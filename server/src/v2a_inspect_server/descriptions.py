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
) -> list[GenerationGroup]:
    events_by_id = {event.event_id: event for event in sound_events}
    ambience_by_id = {ambience.ambience_id: ambience for ambience in ambience_beds}
    sources_by_id = {source.source_id: source for source in physical_sources}
    updated: list[GenerationGroup] = []

    for group in generation_groups:
        if group.member_event_ids:
            member_events = [events_by_id[event_id] for event_id in group.member_event_ids if event_id in events_by_id]
            source_labels = []
            event_types = []
            materials = []
            patterns = []
            for event in member_events:
                source = sources_by_id.get(event.source_id)
                if source and source.label_candidates:
                    source_labels.append(source.label_candidates[0].label)
                event_types.append(event.event_type)
                if event.material_or_surface:
                    materials.append(event.material_or_surface)
                if event.pattern:
                    patterns.append(event.pattern)
            label = _majority_or_default(source_labels, default="source")
            event_type = _majority_or_default(event_types, default="sound_event")
            material = _majority_or_default(materials, default="generic")
            pattern = _majority_or_default(patterns, default="mixed")
            canonical_description = f"{label} {event_type.replace('_', ' ')} on {material} with {pattern} texture"
            updated_group = group.model_copy(
                update={
                    "canonical_description": canonical_description,
                    "description_confidence": round(group.group_confidence, 4),
                    "description_rationale": "structured synthesis from source labels, event types, materials, and patterns",
                }
            )
            if description_writer is not None:
                draft = description_writer.write_group_description(
                    _event_group_context(
                        group=updated_group,
                        member_events=member_events,
                        physical_sources=physical_sources,
                    )
                )
                if draft is not None:
                    updated_group = updated_group.model_copy(
                        update={
                            "canonical_description": draft.canonical_description,
                            "description_confidence": round(
                                draft.description_confidence, 4
                            ),
                            "description_rationale": draft.description_rationale,
                        }
                    )
            updated.append(updated_group)
            continue

        member_ambience = [ambience_by_id[ambience_id] for ambience_id in group.member_ambience_ids if ambience_id in ambience_by_id]
        environment = _majority_or_default([ambience.environment_type for ambience in member_ambience], default="environment")
        texture = _majority_or_default([ambience.acoustic_profile for ambience in member_ambience], default="continuous ambience")
        updated_group = group.model_copy(
            update={
                "canonical_description": f"{environment} ambience bed with {texture}",
                "description_confidence": round(group.group_confidence, 4),
                "description_rationale": "structured ambience synthesis from environment type and acoustic profile",
            }
        )
        if description_writer is not None:
            draft = description_writer.write_group_description(
                _ambience_group_context(
                    group=updated_group,
                    member_ambience=member_ambience,
                )
            )
            if draft is not None:
                updated_group = updated_group.model_copy(
                    update={
                        "canonical_description": draft.canonical_description,
                        "description_confidence": round(
                            draft.description_confidence, 4
                        ),
                        "description_rationale": draft.description_rationale,
                    }
                )
        updated.append(updated_group)

    return updated


def _event_group_context(
    *,
    group: GenerationGroup,
    member_events: list[SoundEventSegment],
    physical_sources: list[PhysicalSourceTrack],
) -> dict[str, object]:
    sources_by_id = {source.source_id: source for source in physical_sources}
    return {
        "group_id": group.group_id,
        "canonical_label": group.canonical_label,
        "event_count": len(member_events),
        "source_labels": [
            [
                candidate.label
                for candidate in sources_by_id.get(event.source_id).label_candidates[:3]
            ]
            if event.source_id in sources_by_id
            else []
            for event in member_events
        ],
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "material_or_surface": event.material_or_surface,
                "intensity": event.intensity,
                "texture": event.texture,
                "pattern": event.pattern,
                "confidence": event.confidence,
            }
            for event in member_events
        ],
        "route_prior": group.route_decision.model_type,
    }


def _ambience_group_context(
    *,
    group: GenerationGroup,
    member_ambience: list[AmbienceBed],
) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "canonical_label": group.canonical_label,
        "ambience_segments": [
            {
                "ambience_id": ambience.ambience_id,
                "environment_type": ambience.environment_type,
                "acoustic_profile": ambience.acoustic_profile,
                "confidence": ambience.confidence,
            }
            for ambience in member_ambience
        ],
        "route_prior": group.route_decision.model_type,
    }


def _majority_or_default(values: list[str], *, default: str) -> str:
    if not values:
        return default
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
