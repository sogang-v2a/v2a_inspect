from __future__ import annotations

import math

from .types import GroupRoutingDecision, Sam3EntityTrack, TrackRoutingDecision


def route_track(track: Sam3EntityTrack) -> TrackRoutingDecision:
    eventfulness = (
        track.features.motion_score + track.features.interaction_score
    ) / 2.0
    complexity = max(track.features.crowd_score, track.features.camera_dynamics_score)
    persistence = min(max(track.end_seconds - track.start_seconds, 0.0) / 5.0, 1.0)

    vta_bias = eventfulness * 0.55 + persistence * 0.35
    tta_bias = (
        complexity * 0.65
        + (1.0 - eventfulness) * 0.15
        + track.features.crowd_score * 0.2
    )
    margin = vta_bias - tta_bias

    if margin >= 0.05:
        model_type = "VTA"
        confidence = min(0.55 + margin, 0.95)
    else:
        model_type = "TTA"
        confidence = min(0.55 + abs(margin), 0.95)

    reasoning = (
        "Visual routing from motion/interactions/crowd/camera dynamics only; "
        f"eventfulness={eventfulness:.2f}, complexity={complexity:.2f}, persistence={persistence:.2f}."
    )
    return TrackRoutingDecision(
        track_id=track.track_id,
        model_type=model_type,
        confidence=round(confidence, 3),
        reasoning=reasoning,
    )


def aggregate_group_routes(
    group_id: str,
    member_track_ids: list[str],
    decisions_by_track_id: dict[str, TrackRoutingDecision],
) -> GroupRoutingDecision:
    member_decisions = [
        decisions_by_track_id[track_id]
        for track_id in member_track_ids
        if track_id in decisions_by_track_id
    ]
    if not member_decisions:
        return GroupRoutingDecision(
            group_id=group_id,
            model_type="TTA",
            confidence=0.5,
            member_track_ids=member_track_ids,
            reasoning="No track-level routing decisions were available.",
        )

    votes = {"TTA": 0, "VTA": 0}
    for decision in member_decisions:
        votes[decision.model_type] += 1
    model_type = "VTA" if votes["VTA"] > votes["TTA"] else "TTA"

    selected_confidences = [
        max(decision.confidence, 1e-6)
        for decision in member_decisions
        if decision.model_type == model_type
    ]
    if not selected_confidences:
        selected_confidences = [
            max(decision.confidence, 1e-6) for decision in member_decisions
        ]

    geometric_mean = math.prod(selected_confidences) ** (1 / len(selected_confidences))
    return GroupRoutingDecision(
        group_id=group_id,
        model_type=model_type,
        confidence=round(min(max(geometric_mean, 0.0), 0.95), 3),
        member_track_ids=member_track_ids,
        reasoning="Aggregated from track routing decisions with majority vote and geometric-mean confidence.",
    )
