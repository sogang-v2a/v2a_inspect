from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from v2a_inspect.tools.types import FrameBatch, SampledFrame, Sam3EntityTrack, Sam3TrackPoint
from v2a_inspect_server.crops import crop_tracks, group_crop_paths_by_track


class CropTrackTests(unittest.TestCase):
    def test_crop_tracks_uses_bbox_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame_path = root / "frame.jpg"
            Image.new("RGB", (64, 64), color="white").save(frame_path)
            batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path=str(frame_path))
                    ],
                )
            ]
            tracks = [
                Sam3EntityTrack(
                    track_id="trk0",
                    scene_index=0,
                    start_seconds=0.0,
                    end_seconds=0.0,
                    confidence=0.9,
                    points=[
                        Sam3TrackPoint(timestamp_seconds=0.0, bbox_xyxy=[8, 8, 32, 32])
                    ],
                )
            ]
            crops = crop_tracks(batches, tracks, output_dir=str(root / "crops"))
            self.assertEqual(len(crops), 1)
            self.assertEqual(crops[0].track_id, "trk0")
            self.assertTrue(Path(crops[0].crop_path).exists())

    def test_crop_tracks_can_decode_simple_mask_rle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame_path = root / "frame.jpg"
            Image.new("RGB", (4, 4), color="white").save(frame_path)
            mask_payload = json.dumps({"size": [4, 4], "counts": [5, 2, 9]})
            batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path=str(frame_path))
                    ],
                )
            ]
            tracks = [
                Sam3EntityTrack(
                    track_id="trk0",
                    scene_index=0,
                    start_seconds=0.0,
                    end_seconds=0.0,
                    confidence=0.9,
                    points=[
                        Sam3TrackPoint(timestamp_seconds=0.0, mask_rle=mask_payload)
                    ],
                )
            ]
            crops = crop_tracks(batches, tracks, output_dir=str(root / "crops"))
            grouped = group_crop_paths_by_track(crops)
        self.assertEqual(len(crops), 1)
        self.assertIn("trk0", grouped)
        self.assertEqual(len(grouped["trk0"]), 1)


if __name__ == "__main__":
    unittest.main()
