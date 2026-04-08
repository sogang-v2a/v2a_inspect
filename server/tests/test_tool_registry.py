from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from v2a_inspect.tools.types import FrameBatch, SampledFrame, Sam3EntityTrack, Sam3TrackPoint
from v2a_inspect_server.tool_registry import build_tool_registry


class ToolRegistryTests(unittest.TestCase):
    def test_registry_exposes_direct_tool_surface(self) -> None:
        fake_runtime = SimpleNamespace(
            sam3_client=SimpleNamespace(
                extract_entities=lambda frame_batches, prompts_by_scene=None: {"tracks": []},
                recover_with_text_prompt=lambda frame_batches, text_prompt: {"tracks": []},
            ),
            embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: []),
            label_client=SimpleNamespace(score_image_labels=lambda image_paths, labels: []),
        )
        registry = build_tool_registry(fake_runtime)
        self.assertIn("structural_overview", registry)
        self.assertIn("recover_with_text_prompt", registry)
        self.assertIn("validate_bundle", registry)

    def test_crop_and_embedding_tools_are_callable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "frame.jpg"
            from PIL import Image
            Image.new("RGB", (64, 64), color="white").save(frame_path)
            fake_runtime = SimpleNamespace(
                sam3_client=SimpleNamespace(
                    extract_entities=lambda frame_batches, prompts_by_scene=None: {"tracks": []},
                    recover_with_text_prompt=lambda frame_batches, text_prompt: {"tracks": []},
                ),
                embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: track_image_paths),
                label_client=SimpleNamespace(score_image_labels=lambda image_paths, labels: []),
            )
            registry = build_tool_registry(fake_runtime)
            frame_batches = [FrameBatch(scene_index=0, frames=[SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path=str(frame_path))])]
            tracks = [Sam3EntityTrack(track_id="trk0", scene_index=0, start_seconds=0.0, end_seconds=0.0, confidence=0.9, points=[Sam3TrackPoint(timestamp_seconds=0.0, bbox_xyxy=[0, 0, 10, 10])])]
            track_crops = registry["crop_tracks"].handler(frame_batches=frame_batches, tracks=tracks, output_dir=str(Path(tmp_dir) / "crops"))
            self.assertEqual(len(track_crops), 1)
            embedded = registry["embed_track_crops"].handler(track_image_paths={"trk0": [track_crops[0].crop_path]})
            self.assertEqual(embedded, {"trk0": [track_crops[0].crop_path]})


if __name__ == "__main__":
    unittest.main()
