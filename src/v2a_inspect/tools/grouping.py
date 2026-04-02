from __future__ import annotations

import math

from .types import CandidateGroup, CandidateGroupSet, EntityEmbedding, Sam3EntityTrack


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=False)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def group_entity_embeddings(
    embeddings: list[EntityEmbedding],
    *,
    tracks_by_id: dict[str, Sam3EntityTrack] | None = None,
    threshold: float = 0.86,
    same_scene_threshold: float = 0.92,
) -> CandidateGroupSet:
    visited: set[str] = set()
    groups: list[CandidateGroup] = []

    for index, embedding in enumerate(embeddings):
        if embedding.track_id in visited:
            continue

        member_ids = [embedding.track_id]
        visited.add(embedding.track_id)
        confidence_scores: list[float] = [1.0]

        for other in embeddings[index + 1 :]:
            if other.track_id in visited:
                continue
            similarity = cosine_similarity(embedding.vector, other.vector)
            required_threshold = _required_threshold(
                embedding.track_id,
                other.track_id,
                tracks_by_id=tracks_by_id,
                threshold=threshold,
                same_scene_threshold=same_scene_threshold,
            )
            if similarity >= required_threshold:
                member_ids.append(other.track_id)
                visited.add(other.track_id)
                confidence_scores.append(similarity)

        groups.append(
            CandidateGroup(
                group_id=f"cg{len(groups)}",
                member_track_ids=member_ids,
                confidence=round(sum(confidence_scores) / len(confidence_scores), 3),
            )
        )

    return CandidateGroupSet(groups=groups)


def _required_threshold(
    left_track_id: str,
    right_track_id: str,
    *,
    tracks_by_id: dict[str, Sam3EntityTrack] | None,
    threshold: float,
    same_scene_threshold: float,
) -> float:
    if tracks_by_id is None:
        return threshold
    left_track = tracks_by_id.get(left_track_id)
    right_track = tracks_by_id.get(right_track_id)
    if left_track is None or right_track is None:
        return threshold
    if left_track.scene_index == right_track.scene_index:
        return same_scene_threshold
    return threshold
