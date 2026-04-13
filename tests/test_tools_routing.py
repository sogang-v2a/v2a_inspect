from __future__ import annotations

import unittest

import v2a_inspect.tools.routing as routing
from v2a_inspect.tools import Sam3EntityTrack, Sam3VisualFeatures


class RoutingTests(unittest.TestCase):
    def test_tracks_still_model_without_routing_helpers(self) -> None:
        track = Sam3EntityTrack(
            track_id="t0",
            scene_index=0,
            start_seconds=0.0,
            end_seconds=4.0,
            features=Sam3VisualFeatures(motion_score=0.95),
        )
        self.assertEqual(track.track_id, "t0")
        self.assertFalse(hasattr(routing, "route_track"))


if __name__ == "__main__":
    unittest.main()
