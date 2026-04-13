from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from v2a_inspect.contracts import (
    AmbienceBed,
    GenerationGroup,
    LabelCandidate,
    PhysicalSourceTrack,
    RoutingDecision,
    SoundEventSegment,
)
from v2a_inspect.tools import aggregate_group_routes
from v2a_inspect.tools.types import CandidateGroup, Sam3EntityTrack, TrackRoutingDecision


def build_sound_event_segments(
    physical_sources: list[PhysicalSourceTrack],
    *,
    tracks_by_id: dict[str, Sam3EntityTrack],
) -> list[SoundEventSegment]:
    segments: list[SoundEventSegment] = []
    for source in physical_sources:
        source_track_ids = [track_id for track_id in source.track_refs if track_id in tracks_by_id]
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
                        "strong" if track.features.motion_score >= 0.7 else "light"
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
    evidence_windows: Sequence[object],
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
                overlap = max(
                    min(span_end, end_time) - max(span_start, start_time), 0.0
                )
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
    candidate_groups: list[CandidateGroup] | None = None,
    routing_decisions_by_track: dict[str, TrackRoutingDecision] | None = None,
    scene_hypotheses_by_window: dict[int, dict[str, object]] | None = None,
    proposal_provenance_by_window: dict[int, dict[str, object]] | None = None,
) -> list[GenerationGroup]:
    labels_by_source = {
        source.source_id: source.label_candidates[0].label
        for source in physical_sources
        if source.label_candidates
    }
    track_ids_by_source = {
        source.source_id: list(source.track_refs)
        for source in physical_sources
    }
    candidate_groups_by_track: dict[str, CandidateGroup] = {}
    for candidate_group in candidate_groups or []:
        for track_id in candidate_group.member_track_ids:
            candidate_groups_by_track[track_id] = candidate_group
    grouped_events: dict[str, list[SoundEventSegment]] = defaultdict(list)
    for event in sound_events:
        grouped_events[_event_group_key(
            event,
            labels_by_source.get(event.source_id),
            scene_hypotheses_by_window=scene_hypotheses_by_window,
            proposal_provenance_by_window=proposal_provenance_by_window,
            track_ids_by_source=track_ids_by_source,
        )].append(event)

    groups: list[GenerationGroup] = []
    for index, (group_key, events) in enumerate(grouped_events.items()):
        member_track_ids = sorted(
            {
                track_id
                for event in events
                for track_id in track_ids_by_source.get(event.source_id, [])
            }
        )
        route_decision = _provisional_route(
            event_type=events[0].event_type, ambience=False
        )
        routing_candidates = []
        if member_track_ids and routing_decisions_by_track:
            aggregated_route = aggregate_group_routes(
                f"gen-{index:04d}",
                member_track_ids,
                routing_decisions_by_track,
            )
            routing_candidates = [aggregated_route.model_dump(mode="json")]
            route_decision = RoutingDecision(
                model_type=aggregated_route.model_type,
                confidence=aggregated_route.confidence,
                factors=["group_routing_priors", aggregated_route.aggregate_method],
                reasoning=aggregated_route.reasoning,
                rule_based=True,
            )
        candidate_group_ids = sorted(
            {
                candidate_groups_by_track[track_id].group_id
                for track_id in member_track_ids
                if track_id in candidate_groups_by_track
            }
        )
        groups.append(
            GenerationGroup(
                group_id=f"gen-{index:04d}",
                member_event_ids=[event.event_id for event in events],
                canonical_label=group_key,
                canonical_description=f"provisional acoustic recipe for {group_key}",
                group_confidence=round(
                    sum(event.confidence for event in events) / len(events), 4
                ),
                route_decision=route_decision,
                reasoning_summary=(
                    "heuristic acoustic-equivalence grouping informed by candidate embedding groups"
                    if candidate_group_ids
                    else "acoustic recipe grouping from source identity, interaction texture, materials, and scene hypotheses"
                ),
                routing_candidates=routing_candidates,
                temporary_adapter_from=(
                    "candidate_groups" if candidate_group_ids else None
                ),
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
                route_decision=_provisional_route(
                    event_type=ambience.environment_type, ambience=True
                ),
                reasoning_summary="ambience beds are grouped by environment profile rather than source identity",
            )
        )
    return groups


def _event_type_from_track(track: Sam3EntityTrack) -> str:
    if (
        track.features.motion_score >= 0.7
        and track.features.interaction_score >= 0.6
    ):
        return "active_interaction"
    if (
        track.features.motion_score >= 0.6
        and track.features.continuity_score >= 0.5
    ):
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


def _event_group_key(
    event: SoundEventSegment,
    source_label: str | None,
    *,
    scene_hypotheses_by_window: dict[int, dict[str, object]] | None,
    proposal_provenance_by_window: dict[int, dict[str, object]] | None,
    track_ids_by_source: dict[str, list[str]],
) -> str:
    label = source_label or "unknown"
    material = event.material_or_surface or "generic"
    interaction = sorted(event.interaction_flags or [])
    motion = event.motion_profile or "unknown"
    texture = event.texture or "generic"
    scene_key = _scene_recipe_hint(
        event.source_id,
        track_ids_by_source=track_ids_by_source,
        scene_hypotheses_by_window=scene_hypotheses_by_window,
        proposal_provenance_by_window=proposal_provenance_by_window,
    )
    return f"{label}:{event.event_type}:{material}:{motion}:{texture}:{'+'.join(interaction) or 'none'}:{scene_key}"


def _scene_recipe_hint(
    source_id: str,
    *,
    track_ids_by_source: dict[str, list[str]],
    scene_hypotheses_by_window: dict[int, dict[str, object]] | None,
    proposal_provenance_by_window: dict[int, dict[str, object]] | None,
) -> str:
    candidate_scene_indices = sorted(
        {
            int(track_id.split("-")[1])
            for track_id in track_ids_by_source.get(source_id, [])
            if "-" in track_id and track_id.split("-")[1].isdigit()
        }
    )
    for scene_index in candidate_scene_indices:
        hypothesis = (scene_hypotheses_by_window or {}).get(scene_index, {})
        if hypothesis:
            cues = [
                *list(hypothesis.get("background_environment", [])[:1]),
                *list(hypothesis.get("material_cues", [])[:1]),
                *list(hypothesis.get("interactions", [])[:1]),
            ]
            deduped = [cue for cue in cues if cue]
            if deduped:
                return "|".join(deduped)
        provenance = (proposal_provenance_by_window or {}).get(scene_index, {})
        ontology_hints = list(provenance.get("ontology_semantics", [])[:2])
        if ontology_hints:
            return "|".join(str(item) for item in ontology_hints)
    return "generic_scene"


def _provisional_route(*, event_type: str, ambience: bool) -> RoutingDecision:
    if ambience:
        return RoutingDecision(
            model_type="TTA",
            confidence=0.9,
            factors=["ambience_bed", event_type],
            reasoning="ambience beds default to text-to-audio for stable background generation",
            rule_based=True,
        )
    model_type = (
        "VTA" if event_type in {"continuous_motion", "active_interaction"} else "TTA"
    )
    return RoutingDecision(
        model_type=model_type,
        confidence=0.7,
        factors=[event_type],
        reasoning="stage-4 provisional route from event texture heuristic",
        rule_based=True,
    )
