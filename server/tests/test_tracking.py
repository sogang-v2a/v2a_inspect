from __future__ import annotations

import unittest

from v2a_inspect.tools.types import FrameBatch, Sam3VisualFeatures, SampledFrame
from v2a_inspect_server.tracking import FrameDetection, link_frame_detections


class TrackingTests(unittest.TestCase):
    def test_link_frame_detections_builds_temporal_track_points(self) -> None:
        batch = FrameBatch(
            scene_index=0,
            frames=[
                SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path="/tmp/f0.jpg"),
                SampledFrame(scene_index=0, timestamp_seconds=0.5, image_path="/tmp/f1.jpg"),
            ],
        )
        detections_by_frame = [
            [
                FrameDetection(
                    frame=batch.frames[0],
                    bbox_xyxy=[10.0, 10.0, 30.0, 30.0],
                    confidence=0.9,
                    label_hint="cat",
                )
            ],
            [
                FrameDetection(
                    frame=batch.frames[1],
                    bbox_xyxy=[11.0, 10.0, 31.0, 30.0],
                    confidence=0.8,
                    label_hint="cat",
                )
            ],
        ]

        tracks = link_frame_detections(
            batch,
            detections_by_frame=detections_by_frame,
            features=Sam3VisualFeatures(),
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].points), 2)
        self.assertEqual(tracks[0].points[0].frame_path, "/tmp/f0.jpg")
        self.assertEqual(tracks[0].points[1].frame_path, "/tmp/f1.jpg")
        self.assertEqual(tracks[0].label_hint, "cat")

    def test_link_frame_detections_filters_single_low_confidence_observations(self) -> None:
        batch = FrameBatch(
            scene_index=0,
            frames=[
                SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path="/tmp/f0.jpg"),
            ],
        )
        detections_by_frame = [
            [
                FrameDetection(
                    frame=batch.frames[0],
                    bbox_xyxy=[10.0, 10.0, 30.0, 30.0],
                    confidence=0.4,
                    label_hint="cat",
                )
            ],
        ]

        tracks = link_frame_detections(
            batch,
            detections_by_frame=detections_by_frame,
            features=Sam3VisualFeatures(),
        )

        self.assertEqual(tracks, [])


if __name__ == "__main__":
    unittest.main()
