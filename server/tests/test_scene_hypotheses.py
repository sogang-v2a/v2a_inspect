from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from v2a_inspect.tools.types import FrameBatch, SampledFrame
from v2a_inspect_server.scene_hypotheses import propose_moving_regions


class SceneHypothesesTests(unittest.TestCase):
    def test_propose_moving_regions_extracts_changed_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            left_path = Path(tmp_dir) / "left.jpg"
            right_path = Path(tmp_dir) / "right.jpg"
            Image.new("RGB", (64, 64), color="black").save(left_path)
            image = Image.new("RGB", (64, 64), color="black")
            for x in range(20, 36):
                for y in range(18, 34):
                    image.putpixel((x, y), (255, 255, 255))
            image.save(right_path)
            frame_batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path=str(left_path)),
                        SampledFrame(scene_index=0, timestamp_seconds=0.5, image_path=str(right_path)),
                    ],
                )
            ]
            proposals = propose_moving_regions(
                frame_batches,
                output_root=str(Path(tmp_dir) / "motion"),
                threshold=0.01,
            )
            self.assertEqual(len(proposals[0]), 1)
            self.assertGreater(proposals[0][0].motion_score, 0.0)
            self.assertTrue(Path(proposals[0][0].crop_path).exists())

    def test_propose_moving_regions_returns_empty_for_identical_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            left_path = Path(tmp_dir) / "left.jpg"
            right_path = Path(tmp_dir) / "right.jpg"
            Image.new("RGB", (64, 64), color="black").save(left_path)
            Image.new("RGB", (64, 64), color="black").save(right_path)
            frame_batches = [
                FrameBatch(
                    scene_index=0,
                    frames=[
                        SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path=str(left_path)),
                        SampledFrame(scene_index=0, timestamp_seconds=0.5, image_path=str(right_path)),
                    ],
                )
            ]
            proposals = propose_moving_regions(frame_batches, output_root=str(Path(tmp_dir) / "motion"))
            self.assertEqual(proposals[0], [])


if __name__ == "__main__":
    unittest.main()
