from __future__ import annotations

from collections import defaultdict

from v2a_inspect.contracts import (
    AmbienceBed,
    GenerationGroup,
    LabelCandidate,
    PhysicalSourceTrack,
    RoutingDecision,
    SoundEventSegment,
)
from v2a_inspect.tools.types import Sam3EntityTrack


def build_sound_event_segments(
    physical_sources: list[PhysicalSourceTrack],
    *,
    tracks_by_id: dict[str, Sam3EntityTrack],
) -> list[SoundEventSegment]:
    segments: list[SoundEventSegment] = []
    for source in physical_sources:
        source_track_ids = [ref for ref in source.evidence_refs if ref in tracks_by_id]
        if not source_track_ids:
            for span_index, span in enumerate(source.spans):
                segments.append(
                    SoundEventSegment(
                        event_id=f"{source.source_id}-event-{span_index:02d}",
                        source_id=source.source_id,
                        start_time=span[0],
                        end_time=span[1],
                        event_type="presence_texture",
                        sync_strength=source.identity_confidence,
                        motion_profile="unknown",
                        confidence=source.identity_confidence,
                    )
                )
            continue

        for event_index, track_id in enumerate(source_track_ids):
            track = tracks_by_id[track_id]
            event_type = _event_type_from_track(track)
            material_hint = _material_hint(source.label_candidates)
            segments.append(
                SoundEventSegment(
                    event_id=f"{source.source_id}-event-{event_index:02d}",
                    source_id=source.source_id,
                    start_time=track.start_seconds,
                    end_time=track.end_seconds,
                    event_type=event_type,
                    sync_strength=min(max(track.features.interaction_score, 0.0), 1.0),
                    motion_profile=(
                        "high_motion"
                        if track.features.motion_score >= 0.7
                        else "low_motion"
                    ),
                    interaction_flags=(
                        ["contact_like"]
                        if track.features.interaction_score >= 0.6
                        else []
                    ),
                    material_or_surface=material_hint,
                    intensity=(
                        "strong"
                        if track.features.motion_score >= 0.7
                        else "light"
                    ),
                    texture=(
                        "continuous"
                        if event_type in {"continuous_motion", "active_interaction"}
                        else "punctate"
                    ),
                    pattern=(
                        "repeating"
                        if track.features.motion_score >= 0.7
                        else "sporadic"
                    ),
                    confidence=round(track.confidence, 4),
                )
            )
    return segments


def build_ambience_beds(
    evidence_windows: list[object],
    physical_sources: list[PhysicalSourceTrack],
) -> list[AmbienceBed]:
    beds: list[AmbienceBed] = []
    for index, window in enumerate(evidence_windows):
        start_time = float(getattr(window, "start_time", 0.0))
        end_time = float(getattr(window, "end_time", start_time))
        window_duration = max(end_time - start_time, 0.0)
        if window_duration <= 0.0:
            continue
        covered = 0.0
        for source in physical_sources:
            for span_start, span_end in source.spans:
                overlap = max(min(span_end, end_time) - max(span_start, start_time), 0.0)
                covered += overlap
        coverage_ratio = min(covered / window_duration, 1.0)
        if coverage_ratio < 0.6:
            beds.append(
                AmbienceBed(
                    ambience_id=f"ambience-{index:04d}",
                    start_time=start_time,
                    end_time=end_time,
                    environment_type="scene_bed",
                    acoustic_profile="continuous visual environment texture",
                    foreground_exclusion_notes="heuristic ambience bed for uncovered structural window",
                    confidence=round(1.0 - coverage_ratio, 4),
                )
            )
    return beds


def build_generation_groups(
    sound_events: list[SoundEventSegment],
    ambience_beds: list[AmbienceBed],
    *,
    physical_sources: list[PhysicalSourceTrack],
) -> list[GenerationGroup]:
    labels_by_source = {
        source.source_id: source.label_candidates[0].label
        for source in physical_sources
        if source.label_candidates
    }
    grouped_events: dict[str, list[SoundEventSegment]] = defaultdict(list)
    for event in sound_events:
        grouped_events[_event_group_key(event, labels_by_source.get(event.source_id))].append(event)

    groups: list[GenerationGroup] = []
    for index, (group_key, events) in enumerate(grouped_events.items()):
        groups.append(
            GenerationGroup(
                group_id=f"gen-{index:04d}",
                member_event_ids=[event.event_id for event in events],
                canonical_label=group_key,
                canonical_description=f"provisional acoustic recipe for {group_key}",
                group_confidence=round(sum(event.confidence for event in events) / len(events), 4),
                route_decision=_provisional_route(event_type=events[0].event_type, ambience=False),
                reasoning_summary="heuristic acoustic-equivalence grouping from event type and source label",
            )
        )
    offset = len(groups)
    for index, ambience in enumerate(ambience_beds):
        groups.append(
            GenerationGroup(
                group_id=f"gen-{offset + index:04d}",
                member_ambience_ids=[ambience.ambience_id],
                canonical_label=f"ambience:{ambience.environment_type}",
                canonical_description=ambience.acoustic_profile,
                group_confidence=ambience.confidence,
                route_decision=_provisional_route(event_type=ambience.environment_type, ambience=True),
                reasoning_summary="ambience beds are grouped by environment profile rather than source identity",
            )
        )
    return groups


def _event_type_from_track(track: Sam3EntityTrack) -> str:
    if track.features.motion_score >= 0.7 and track.features.interaction_score >= 0.6:
        return "active_interaction"
    if track.features.motion_score >= 0.7:
        return "continuous_motion"
    if track.features.interaction_score >= 0.6:
        return "contact_event"
    return "presence_texture"


def _material_hint(label_candidates: list[LabelCandidate]) -> str | None:
    if not label_candidates:
        return None
    top_label = label_candidates[0].label
    if top_label in {"car", "vehicle"}:
        return "metal"
    if top_label in {"person", "animal"}:
        return "ground_contact"
    return None


def _event_group_key(event: SoundEventSegment, source_label: str | None) -> str:
    label = source_label or "unknown"
    material = event.material_or_surface or "generic"
    return f"{label}:{event.event_type}:{material}"


def _provisional_route(*, event_type: str, ambience: bool) -> RoutingDecision:
    if ambience:
        return RoutingDecision(
            model_type="VTA",
            confidence=0.9,
            factors=["ambience_bed", event_type],
            reasoning="ambience beds prefer continuous video-to-audio generation",
            rule_based=True,
        )
    model_type = "VTA" if event_type in {"continuous_motion", "active_interaction"} else "TTA"
    return RoutingDecision(
        model_type=model_type,
        confidence=0.7,
        factors=[event_type],
        reasoning="stage-4 provisional route from event texture heuristic",
        rule_based=True,
    )
