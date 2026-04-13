from __future__ import annotations

import unittest

from v2a_inspect.contracts import LabelCandidate, TrackCrop
from v2a_inspect.tools.types import EntityEmbedding, Sam3EntityTrack
from v2a_inspect_server.reid import (
    build_identity_edges,
    build_provisional_source_tracks,
)


class ReidTests(unittest.TestCase):
    def test_same_window_merges_are_stricter_than_cross_window(self) -> None:
        tracks = [
            Sam3EntityTrack(
                track_id="a",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.9,
            ),
            Sam3EntityTrack(
                track_id="b",
                scene_index=0,
                start_seconds=0.2,
                end_seconds=1.2,
                confidence=0.9,
            ),
            Sam3EntityTrack(
                track_id="c",
                scene_index=1,
                start_seconds=2.0,
                end_seconds=3.0,
                confidence=0.9,
            ),
        ]
        embeddings = [
            EntityEmbedding(track_id="a", model_name="fake", vector=[1.0, 0.0]),
            EntityEmbedding(track_id="b", model_name="fake", vector=[0.7, 0.3]),
            EntityEmbedding(track_id="c", model_name="fake", vector=[1.0, 0.0]),
        ]
        label_candidates = {
            "a": [LabelCandidate(label="person", score=0.9)],
            "b": [LabelCandidate(label="person", score=0.9)],
            "c": [LabelCandidate(label="person", score=0.9)],
        }
        edges = build_identity_edges(
            tracks, embeddings, label_candidates_by_track=label_candidates
        )
        accepted = {
            (edge.source_track_id, edge.target_track_id): edge.accepted
            for edge in edges
        }
        self.assertFalse(accepted[("a", "b")])
        self.assertTrue(accepted[("a", "c")])

    def test_build_provisional_source_tracks_preserves_ambiguity(self) -> None:
        tracks = [
            Sam3EntityTrack(
                track_id="a",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.9,
            ),
            Sam3EntityTrack(
                track_id="c",
                scene_index=1,
                start_seconds=2.0,
                end_seconds=3.0,
                confidence=0.9,
            ),
            Sam3EntityTrack(
                track_id="d",
                scene_index=1,
                start_seconds=2.2,
                end_seconds=3.2,
                confidence=0.7,
            ),
        ]
        embeddings = [
            EntityEmbedding(track_id="a", model_name="fake", vector=[1.0, 0.0]),
            EntityEmbedding(track_id="c", model_name="fake", vector=[1.0, 0.0]),
            EntityEmbedding(track_id="d", model_name="fake", vector=[0.3, 0.7]),
        ]
        label_candidates = {
            "a": [LabelCandidate(label="person", score=0.9)],
            "c": [LabelCandidate(label="person", score=0.9)],
            "d": [LabelCandidate(label="vehicle", score=0.9)],
        }
        edges = build_identity_edges(
            tracks, embeddings, label_candidates_by_track=label_candidates
        )
        crops = [
            TrackCrop(
                crop_id="crop-a",
                track_id="a",
                scene_index=0,
                frame_path="/tmp/a.jpg",
                crop_path="/tmp/a-crop.jpg",
                timestamp_seconds=0.0,
                bbox_xyxy=[0, 0, 10, 10],
            ),
            TrackCrop(
                crop_id="crop-c",
                track_id="c",
                scene_index=1,
                frame_path="/tmp/c.jpg",
                crop_path="/tmp/c-crop.jpg",
                timestamp_seconds=2.0,
                bbox_xyxy=[0, 0, 10, 10],
            ),
            TrackCrop(
                crop_id="crop-d",
                track_id="d",
                scene_index=1,
                frame_path="/tmp/d.jpg",
                crop_path="/tmp/d-crop.jpg",
                timestamp_seconds=2.2,
                bbox_xyxy=[0, 0, 10, 10],
            ),
        ]
        sources = build_provisional_source_tracks(
            tracks,
            edges,
            track_crops=crops,
            label_candidates_by_track=label_candidates,
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual({source.source_id for source in sources}, {"source-a", "source-d"})
        self.assertTrue(any(source.identity_confidence > 0.9 for source in sources))
        self.assertTrue(any(source.identity_confidence < 0.9 for source in sources))

    def test_identity_edges_clamp_negative_similarity_to_zero(self) -> None:
        tracks = [
            Sam3EntityTrack(
                track_id="left",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.9,
            ),
            Sam3EntityTrack(
                track_id="right",
                scene_index=1,
                start_seconds=2.0,
                end_seconds=3.0,
                confidence=0.9,
            ),
        ]
        embeddings = [
            EntityEmbedding(track_id="left", model_name="fake", vector=[1.0, 0.0]),
            EntityEmbedding(track_id="right", model_name="fake", vector=[-1.0, 0.0]),
        ]
        edges = build_identity_edges(tracks, embeddings)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].similarity, 0.0)


if __name__ == "__main__":
    unittest.main()
