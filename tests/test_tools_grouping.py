from __future__ import annotations

import unittest

from v2a_inspect.tools import (
    EntityEmbedding,
    Sam3EntityTrack,
    group_entity_embeddings,
)


class GroupingTests(unittest.TestCase):
    def test_groups_similar_embeddings(self) -> None:
        tracks = {
            "t0": Sam3EntityTrack(
                track_id="t0",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
            ),
            "t1": Sam3EntityTrack(
                track_id="t1",
                scene_index=1,
                start_seconds=1.0,
                end_seconds=2.0,
            ),
            "t2": Sam3EntityTrack(
                track_id="t2",
                scene_index=2,
                start_seconds=2.0,
                end_seconds=3.0,
            ),
        }
        grouped = group_entity_embeddings(
            [
                EntityEmbedding(track_id="t0", vector=[1.0, 0.0]),
                EntityEmbedding(track_id="t1", vector=[0.99, 0.01]),
                EntityEmbedding(track_id="t2", vector=[0.0, 1.0]),
            ],
            tracks_by_id=tracks,
            threshold=0.95,
        )

        self.assertEqual(len(grouped.groups), 2)
        member_sets = {tuple(group.member_track_ids) for group in grouped.groups}
        self.assertIn(("t0", "t1"), member_sets)


if __name__ == "__main__":
    unittest.main()
