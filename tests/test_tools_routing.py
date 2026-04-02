from __future__ import annotations

import unittest

from v2a_inspect.tools import (
    Sam3EntityTrack,
    Sam3VisualFeatures,
    aggregate_group_routes,
    route_track,
)


class RoutingTests(unittest.TestCase):
    def test_high_motion_track_prefers_vta(self) -> None:
        track = Sam3EntityTrack(
            track_id="t0",
            scene_index=0,
            start_seconds=0.0,
            end_seconds=4.0,
            features=Sam3VisualFeatures(
                motion_score=0.95,
                interaction_score=0.9,
                crowd_score=0.1,
                camera_dynamics_score=0.2,
            ),
        )
        decision = route_track(track)
        self.assertEqual(decision.model_type, "VTA")

    def test_group_aggregate_uses_majority_vote(self) -> None:
        first = route_track(
            Sam3EntityTrack(
                track_id="t0",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=4.0,
                features=Sam3VisualFeatures(
                    motion_score=0.95,
                    interaction_score=0.9,
                    crowd_score=0.1,
                    camera_dynamics_score=0.2,
                ),
            )
        )
        second = route_track(
            Sam3EntityTrack(
                track_id="t1",
                scene_index=1,
                start_seconds=1.0,
                end_seconds=2.0,
                features=Sam3VisualFeatures(
                    motion_score=0.1,
                    interaction_score=0.2,
                    crowd_score=0.9,
                    camera_dynamics_score=0.8,
                ),
            )
        )
        third = route_track(
            Sam3EntityTrack(
                track_id="t2",
                scene_index=2,
                start_seconds=2.0,
                end_seconds=3.0,
                features=Sam3VisualFeatures(
                    motion_score=0.15,
                    interaction_score=0.2,
                    crowd_score=0.85,
                    camera_dynamics_score=0.75,
                ),
            )
        )

        grouped = aggregate_group_routes(
            "g0",
            ["t0", "t1", "t2"],
            {
                "t0": first,
                "t1": second,
                "t2": third,
            },
        )
        self.assertEqual(grouped.model_type, "TTA")
        self.assertGreater(grouped.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
