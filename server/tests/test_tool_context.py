from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from v2a_inspect.workflows import InspectOptions
from v2a_inspect_server.runtime import ToolingRuntime
from v2a_inspect_server.tool_context import build_tool_context


class ToolContextTests(unittest.TestCase):
    @patch("v2a_inspect_server.tool_context.sample_frames")
    @patch("v2a_inspect_server.tool_context.detect_scenes")
    @patch("v2a_inspect_server.tool_context.probe_video")
    def test_build_tool_context_collects_probe_and_scene_summary(
        self,
        mock_probe_video,
        mock_detect_scenes,
        mock_sample_frames,
    ) -> None:
        mock_probe_video.return_value = type(
            "Probe",
            (),
            {
                "duration_seconds": 3.0,
                "fps": 2.0,
                "width": 320,
                "height": 240,
            },
        )()
        mock_detect_scenes.return_value = [
            type(
                "Scene",
                (),
                {
                    "scene_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                    "strategy": "fixed_window",
                },
            )()
        ]
        mock_sample_frames.return_value = [
            type(
                "Batch",
                (),
                {
                    "scene_index": 0,
                    "frames": [
                        type("Frame", (), {"image_path": "/tmp/frame0.jpg"})(),
                        type("Frame", (), {"image_path": "/tmp/frame1.jpg"})(),
                    ],
                },
            )()
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("v2a_inspect.settings.settings.shared_video_dir", Path(tmp_dir)):
                context = build_tool_context(
                    "/tmp/video.mp4",
                    options=InspectOptions(),
                )

        self.assertIn("video_probe", context)
        self.assertIn("scene_boundaries", context)
        self.assertIn("frame_batches", context)
        self.assertIn(
            "Tool-detected scene windows",
            str(context["tool_scene_summary"]),
        )
        progress_messages = context["progress_messages"]
        if not isinstance(progress_messages, list):
            self.fail("progress_messages should be a list")
        self.assertEqual(len(progress_messages), 3)

    @patch("v2a_inspect_server.tool_context.sample_frames")
    @patch("v2a_inspect_server.tool_context.detect_scenes")
    @patch("v2a_inspect_server.tool_context.probe_video")
    def test_build_tool_context_includes_sam3_hints_when_runtime_available(
        self,
        mock_probe_video,
        mock_detect_scenes,
        mock_sample_frames,
    ) -> None:
        mock_probe_video.return_value = type(
            "Probe",
            (),
            {
                "duration_seconds": 3.0,
                "fps": 2.0,
                "width": 320,
                "height": 240,
            },
        )()
        mock_detect_scenes.return_value = [
            type(
                "Scene",
                (),
                {
                    "scene_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                    "strategy": "fixed_window",
                },
            )()
        ]
        mock_sample_frames.return_value = [
            type(
                "Batch",
                (),
                {
                    "scene_index": 0,
                    "frames": [
                        type("Frame", (), {"image_path": "/tmp/frame0.jpg"})(),
                        type("Frame", (), {"image_path": "/tmp/frame1.jpg"})(),
                    ],
                },
            )()
        ]
        fake_runtime = SimpleNamespace(
            sam3_client=SimpleNamespace(
                extract_entities=lambda _frame_batches: SimpleNamespace(
                    tracks=[
                        SimpleNamespace(
                            track_id="trk0",
                            scene_index=0,
                            start_seconds=0.0,
                            end_seconds=1.0,
                            label_hint="person",
                            confidence=0.9,
                        )
                    ]
                )
            )
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("v2a_inspect.settings.settings.shared_video_dir", Path(tmp_dir)):
                context = build_tool_context(
                    "/tmp/video.mp4",
                    options=InspectOptions(),
                    tooling_runtime=cast(ToolingRuntime, fake_runtime),
                )

        self.assertIn("tool_grouping_hints", context)
        self.assertIn("SAM3 track hints:", str(context["tool_grouping_hints"]))

    @patch("v2a_inspect_server.tool_context.sample_frames")
    @patch("v2a_inspect_server.tool_context.detect_scenes")
    @patch("v2a_inspect_server.tool_context.probe_video")
    def test_build_tool_context_includes_embedding_and_label_hints(
        self,
        mock_probe_video,
        mock_detect_scenes,
        mock_sample_frames,
    ) -> None:
        mock_probe_video.return_value = type(
            "Probe",
            (),
            {
                "duration_seconds": 3.0,
                "fps": 2.0,
                "width": 320,
                "height": 240,
            },
        )()
        mock_detect_scenes.return_value = [
            type(
                "Scene",
                (),
                {
                    "scene_index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                    "strategy": "fixed_window",
                },
            )()
        ]
        mock_sample_frames.return_value = [
            type(
                "Batch",
                (),
                {
                    "scene_index": 0,
                    "frames": [
                        type("Frame", (), {"image_path": "/tmp/frame0.jpg"})(),
                        type("Frame", (), {"image_path": "/tmp/frame1.jpg"})(),
                    ],
                },
            )()
        ]
        fake_runtime = SimpleNamespace(
            sam3_client=SimpleNamespace(
                extract_entities=lambda _frame_batches: SimpleNamespace(
                    tracks=[
                        SimpleNamespace(
                            track_id="trk0",
                            scene_index=0,
                            start_seconds=0.0,
                            end_seconds=1.0,
                            label_hint="person",
                            confidence=0.9,
                        )
                    ]
                )
            ),
            embedding_client=SimpleNamespace(
                embed_images=lambda _image_paths: [
                    SimpleNamespace(
                        track_id="trk0", model_name="dinov2", vector=[1.0, 0.0]
                    )
                ]
            ),
            label_client=SimpleNamespace(
                score_labels=lambda **_kwargs: SimpleNamespace(
                    group_id="cg0",
                    label="person",
                    scores=[
                        SimpleNamespace(label="person", score=0.9),
                        SimpleNamespace(label="object", score=0.1),
                    ],
                )
            ),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("v2a_inspect.settings.settings.shared_video_dir", Path(tmp_dir)):
                context = build_tool_context(
                    "/tmp/video.mp4",
                    options=InspectOptions(),
                    tooling_runtime=cast(ToolingRuntime, fake_runtime),
                )

        self.assertIn("tool_grouping_hints", context)
        hints = str(context["tool_grouping_hints"])
        self.assertIn("Embedding/SAM3 grouping hints:", hints)
        self.assertIn("Label hints:", hints)


if __name__ == "__main__":
    unittest.main()
