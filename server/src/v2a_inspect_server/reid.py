from __future__ import annotations

from collections import defaultdict

from v2a_inspect.contracts import (
    IdentityEdge,
    LabelCandidate,
    PhysicalSourceTrack,
    TrackCrop,
)
from v2a_inspect.tools.grouping import cosine_similarity
from v2a_inspect.tools.types import EntityEmbedding, Sam3EntityTrack


def build_identity_edges(
    tracks: list[Sam3EntityTrack],
    embeddings: list[EntityEmbedding],
    *,
    label_candidates_by_track: dict[str, list[LabelCandidate]] | None = None,
    same_window_threshold: float = 0.995,
    cross_window_threshold: float = 0.92,
) -> list[IdentityEdge]:
    embeddings_by_track = {embedding.track_id: embedding for embedding in embeddings}
    label_candidates_by_track = label_candidates_by_track or {}
    edges: list[IdentityEdge] = []

    for index, left_track in enumerate(tracks):
        for right_track in tracks[index + 1 :]:
            left_embedding = embeddings_by_track.get(left_track.track_id)
            right_embedding = embeddings_by_track.get(right_track.track_id)
            if left_embedding is None or right_embedding is None:
                continue
            similarity = cosine_similarity(
                left_embedding.vector, right_embedding.vector
            )
            same_window = left_track.scene_index == right_track.scene_index
            threshold = same_window_threshold if same_window else cross_window_threshold
            label_compatibility = _label_compatibility(
                label_candidates_by_track.get(left_track.track_id, []),
                label_candidates_by_track.get(right_track.track_id, []),
            )
            confidence = max(
                0.0, min(1.0, (similarity * 0.8) + (label_compatibility * 0.2))
            )
            accepted = confidence >= threshold
            edges.append(
                IdentityEdge(
                    edge_id=f"edge-{left_track.track_id}-{right_track.track_id}",
                    source_track_id=left_track.track_id,
                    target_track_id=right_track.track_id,
                    similarity=round(similarity, 4),
                    same_window=same_window,
                    temporal_gap_seconds=round(
                        abs(right_track.start_seconds - left_track.end_seconds), 3
                    ),
                    label_compatibility=round(label_compatibility, 4),
                    confidence=round(confidence, 4),
                    accepted=accepted,
                    rationale=(
                        "same-window identity requires stricter confidence"
                        if same_window
                        else "cross-window identity candidate"
                    ),
                )
            )
    return edges


def build_provisional_source_tracks(
    tracks: list[Sam3EntityTrack],
    identity_edges: list[IdentityEdge],
    *,
    track_crops: list[TrackCrop],
    label_candidates_by_track: dict[str, list[LabelCandidate]] | None = None,
) -> list[PhysicalSourceTrack]:
    label_candidates_by_track = label_candidates_by_track or {}
    crops_by_track = defaultdict(list)
    for crop in track_crops:
        crops_by_track[crop.track_id].append(crop)

    parent = {track.track_id: track.track_id for track in tracks}

    def find(track_id: str) -> str:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in identity_edges:
        if edge.accepted:
            union(edge.source_track_id, edge.target_track_id)

    groups: dict[str, list[Sam3EntityTrack]] = defaultdict(list)
    for track in tracks:
        groups[find(track.track_id)].append(track)

    accepted_neighbors: dict[str, list[str]] = defaultdict(list)
    for edge in identity_edges:
        if edge.accepted:
            accepted_neighbors[edge.source_track_id].append(edge.target_track_id)
            accepted_neighbors[edge.target_track_id].append(edge.source_track_id)

    physical_sources: list[PhysicalSourceTrack] = []
    for index, member_tracks in enumerate(groups.values()):
        spans = sorted(
            {
                (round(track.start_seconds, 3), round(track.end_seconds, 3))
                for track in member_tracks
            }
        )
        evidence_refs = [track.track_id for track in member_tracks] + [
            crop.crop_id
            for track in member_tracks
            for crop in crops_by_track.get(track.track_id, [])
        ]
        label_candidates = _aggregate_label_candidates(
            label_candidates_by_track,
            [track.track_id for track in member_tracks],
        )
        supporting_edges = [
            edge.confidence
            for edge in identity_edges
            if edge.accepted
            and edge.source_track_id in {track.track_id for track in member_tracks}
            and edge.target_track_id in {track.track_id for track in member_tracks}
        ]
        base_confidence = sum(track.confidence for track in member_tracks) / len(
            member_tracks
        )
        identity_confidence = (
            sum(supporting_edges) / len(supporting_edges)
            if supporting_edges
            else base_confidence
        )
        physical_sources.append(
            PhysicalSourceTrack(
                source_id=f"source-{index:04d}",
                kind="foreground",
                label_candidates=label_candidates,
                spans=spans,
                evidence_refs=evidence_refs,
                identity_confidence=round(identity_confidence, 4),
                reid_neighbors=sorted(
                    {
                        neighbor
                        for track in member_tracks
                        for neighbor in accepted_neighbors.get(track.track_id, [])
                        if neighbor not in {member.track_id for member in member_tracks}
                    }
                ),
                temporary_adapter_from="Sam3EntityTrack",
            )
        )
    return physical_sources


def _label_compatibility(
    left_candidates: list[LabelCandidate],
    right_candidates: list[LabelCandidate],
) -> float:
    if not left_candidates or not right_candidates:
        return 0.5
    left_labels = {candidate.label for candidate in left_candidates[:3]}
    right_labels = {candidate.label for candidate in right_candidates[:3]}
    if left_labels & right_labels:
        return 1.0
    return 0.0


def _aggregate_label_candidates(
    label_candidates_by_track: dict[str, list[LabelCandidate]],
    track_ids: list[str],
) -> list[LabelCandidate]:
    scores: dict[str, list[float]] = defaultdict(list)
    for track_id in track_ids:
        for candidate in label_candidates_by_track.get(track_id, []):
            scores[candidate.label].append(candidate.score)
    aggregated = [
        LabelCandidate(label=label, score=round(sum(values) / len(values), 4))
        for label, values in scores.items()
    ]
    aggregated.sort(key=lambda candidate: candidate.score, reverse=True)
    return aggregated
