from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from v2a_inspect.workflows import InspectOptions
from v2a_inspect_server.runtime import ToolingRuntime
from v2a_inspect_server.tool_context import build_tool_context


class ToolContextTests(unittest.TestCase):
    @patch("v2a_inspect_server.tool_context.hydrate_evidence_windows")
    @patch("v2a_inspect_server.tool_context.export_window_clips")
    @patch("v2a_inspect_server.tool_context.generate_storyboard")
    @patch("v2a_inspect_server.tool_context.sample_frames")
    @patch("v2a_inspect_server.tool_context._frame_output_dir")
    @patch("v2a_inspect_server.tool_context.evidence_windows_to_scene_boundaries")
    @patch("v2a_inspect_server.tool_context.build_evidence_windows")
    @patch("v2a_inspect_server.tool_context.build_candidate_cuts")
    @patch("v2a_inspect_server.tool_context.probe_video")
    def test_build_tool_context_collects_probe_and_scene_summary(
        self,
        mock_probe_video,
        mock_build_candidate_cuts,
        mock_build_evidence_windows,
        mock_scene_boundaries,
        mock_frame_output_dir,
        mock_sample_frames,
        mock_generate_storyboard,
        mock_export_window_clips,
        mock_hydrate_evidence_windows,
    ) -> None:
        mock_probe_video.return_value = SimpleNamespace(
            duration_seconds=3.0,
            fps=2.0,
            width=320,
            height=240,
        )
        mock_build_candidate_cuts.return_value = [
            SimpleNamespace(cut_id="cut-0000", timestamp_seconds=1.5)
        ]
        mock_build_evidence_windows.return_value = [
            SimpleNamespace(
                window_id="window-0000",
                start_time=0.0,
                end_time=3.0,
                cut_refs=["cut-0000"],
                artifact_refs=[],
            )
        ]
        mock_scene_boundaries.return_value = [
            SimpleNamespace(
                scene_index=0,
                start_seconds=0.0,
                end_seconds=3.0,
                strategy="ffmpeg_scene_detect",
            )
        ]
        mock_frame_output_dir.return_value = Path("/tmp/frame-root")
        mock_sample_frames.return_value = [
            SimpleNamespace(
                scene_index=0,
                frames=[
                    SimpleNamespace(image_path="/tmp/frame0.jpg"),
                    SimpleNamespace(image_path="/tmp/frame1.jpg"),
                ],
            )
        ]
        mock_generate_storyboard.return_value = "/tmp/storyboard.jpg"
        mock_export_window_clips.return_value = {"window-0000": "/tmp/window-0000.mp4"}
        mock_hydrate_evidence_windows.return_value = [
            SimpleNamespace(
                window_id="window-0000",
                start_time=0.0,
                end_time=3.0,
                cut_refs=["cut-0000"],
                artifact_refs=["/tmp/storyboard.jpg", "/tmp/window-0000.mp4"],
                sampled_frame_ids=["/tmp/frame0.jpg", "/tmp/frame1.jpg"],
            )
        ]

        context = build_tool_context(
            "/tmp/video.mp4",
            options=InspectOptions(),
        )

        self.assertIn("video_probe", context)
        self.assertIn("candidate_cuts", context)
        self.assertIn("evidence_windows", context)
        self.assertIn("scene_boundaries", context)
        self.assertIn("frame_batches", context)
        self.assertEqual(context["storyboard_path"], "/tmp/storyboard.jpg")
        self.assertIn("Candidate cuts:", str(context["tool_scene_summary"]))
        self.assertIn("Evidence windows:", str(context["tool_scene_summary"]))
        self.assertEqual(len(context["progress_messages"]), 5)

    @patch("v2a_inspect_server.tool_context.hydrate_evidence_windows")
    @patch("v2a_inspect_server.tool_context.export_window_clips")
    @patch("v2a_inspect_server.tool_context.generate_storyboard")
    @patch("v2a_inspect_server.tool_context.sample_frames")
    @patch("v2a_inspect_server.tool_context._frame_output_dir")
    @patch("v2a_inspect_server.tool_context.evidence_windows_to_scene_boundaries")
    @patch("v2a_inspect_server.tool_context.build_evidence_windows")
    @patch("v2a_inspect_server.tool_context.build_candidate_cuts")
    @patch("v2a_inspect_server.tool_context.probe_video")
    def test_build_tool_context_adds_routing_and_verify_hints_for_partial_tracks(
        self,
        mock_probe_video,
        mock_build_candidate_cuts,
        mock_build_evidence_windows,
        mock_scene_boundaries,
        mock_frame_output_dir,
        mock_sample_frames,
        mock_generate_storyboard,
        mock_export_window_clips,
        mock_hydrate_evidence_windows,
    ) -> None:
        mock_probe_video.return_value = SimpleNamespace(
            duration_seconds=3.0,
            fps=2.0,
            width=320,
            height=240,
        )
        mock_build_candidate_cuts.return_value = [
            SimpleNamespace(cut_id="cut-0000", timestamp_seconds=1.5)
        ]
        mock_build_evidence_windows.return_value = [
            SimpleNamespace(window_id="window-0000", start_time=0.0, end_time=3.0, cut_refs=["cut-0000"], artifact_refs=[])
        ]
        mock_scene_boundaries.return_value = [
            SimpleNamespace(scene_index=0, start_seconds=0.0, end_seconds=3.0, strategy="ffmpeg_scene_detect")
        ]
        mock_frame_output_dir.return_value = Path("/tmp/frame-root")
        mock_sample_frames.return_value = [
            SimpleNamespace(
                scene_index=0,
                frames=[
                    SimpleNamespace(image_path="/tmp/frame0.jpg"),
                    SimpleNamespace(image_path="/tmp/frame1.jpg"),
                ],
            )
        ]
        mock_generate_storyboard.return_value = "/tmp/storyboard.jpg"
        mock_export_window_clips.return_value = {}
        mock_hydrate_evidence_windows.return_value = [
            SimpleNamespace(window_id="window-0000", start_time=0.0, end_time=3.0, cut_refs=["cut-0000"], artifact_refs=["/tmp/storyboard.jpg"], sampled_frame_ids=["/tmp/frame0.jpg", "/tmp/frame1.jpg"])
        ]
        fake_runtime = SimpleNamespace(
            sam3_client=SimpleNamespace(
                extract_entities=lambda _frame_batches, **_kwargs: SimpleNamespace(
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

        context = build_tool_context(
            "/tmp/video.mp4",
            options=InspectOptions(),
            tooling_runtime=cast(ToolingRuntime, fake_runtime),
        )

        self.assertIn("tool_grouping_hints", context)
        self.assertIn("tool_routing_hints", context)
        self.assertIn("tool_verify_hints", context)
        self.assertIn("routing_decisions", context)
        self.assertIn("SAM3 track hints:", str(context["tool_grouping_hints"]))
        self.assertIn("Routing/model-selection hints:", str(context["tool_routing_hints"]))
        self.assertIn("track:trk0", str(context["tool_routing_hints"]))
        self.assertIn("Verify/group hints:", str(context["tool_verify_hints"]))
        self.assertIn("priority=low", str(context["tool_verify_hints"]))

    @patch("v2a_inspect_server.tool_context.hydrate_evidence_windows")
    @patch("v2a_inspect_server.tool_context.export_window_clips")
    @patch("v2a_inspect_server.tool_context.generate_storyboard")
    @patch("v2a_inspect_server.tool_context.sample_frames")
    @patch("v2a_inspect_server.tool_context._frame_output_dir")
    @patch("v2a_inspect_server.tool_context.evidence_windows_to_scene_boundaries")
    @patch("v2a_inspect_server.tool_context.build_evidence_windows")
    @patch("v2a_inspect_server.tool_context.build_candidate_cuts")
    @patch("v2a_inspect_server.tool_context.probe_video")
    def test_build_tool_context_includes_embedding_label_and_cross_scene_hints(
        self,
        mock_probe_video,
        mock_build_candidate_cuts,
        mock_build_evidence_windows,
        mock_scene_boundaries,
        mock_frame_output_dir,
        mock_sample_frames,
        mock_generate_storyboard,
        mock_export_window_clips,
        mock_hydrate_evidence_windows,
    ) -> None:
        mock_probe_video.return_value = SimpleNamespace(
            duration_seconds=6.0,
            fps=2.0,
            width=320,
            height=240,
        )
        mock_build_candidate_cuts.return_value = [
            SimpleNamespace(cut_id="cut-0000", timestamp_seconds=3.0),
            SimpleNamespace(cut_id="cut-0001", timestamp_seconds=5.0),
        ]
        mock_build_evidence_windows.return_value = [
            SimpleNamespace(window_id="window-0000", start_time=0.0, end_time=3.0, cut_refs=["cut-0000"], artifact_refs=[]),
            SimpleNamespace(window_id="window-0001", start_time=3.0, end_time=6.0, cut_refs=["cut-0000", "cut-0001"], artifact_refs=[]),
        ]
        mock_scene_boundaries.return_value = [
            SimpleNamespace(scene_index=0, start_seconds=0.0, end_seconds=3.0, strategy="ffmpeg_scene_detect"),
            SimpleNamespace(scene_index=1, start_seconds=3.0, end_seconds=6.0, strategy="ffmpeg_scene_detect"),
        ]
        mock_frame_output_dir.return_value = Path("/tmp/frame-root")
        mock_sample_frames.return_value = [
            SimpleNamespace(scene_index=0, frames=[SimpleNamespace(image_path="/tmp/frame0.jpg")]),
            SimpleNamespace(scene_index=1, frames=[SimpleNamespace(image_path="/tmp/frame1.jpg")]),
        ]
        mock_generate_storyboard.return_value = "/tmp/storyboard.jpg"
        mock_export_window_clips.return_value = {}
        mock_hydrate_evidence_windows.return_value = [
            SimpleNamespace(window_id="window-0000", start_time=0.0, end_time=3.0, cut_refs=["cut-0000"], artifact_refs=["/tmp/storyboard.jpg"], sampled_frame_ids=["/tmp/frame0.jpg"]),
            SimpleNamespace(window_id="window-0001", start_time=3.0, end_time=6.0, cut_refs=["cut-0000", "cut-0001"], artifact_refs=["/tmp/storyboard.jpg"], sampled_frame_ids=["/tmp/frame1.jpg"]),
        ]
        fake_runtime = SimpleNamespace(
            sam3_client=SimpleNamespace(
                extract_entities=lambda _frame_batches, **_kwargs: SimpleNamespace(
                    tracks=[
                        SimpleNamespace(
                            track_id="trk0",
                            scene_index=0,
                            start_seconds=0.0,
                            end_seconds=1.0,
                            label_hint="person",
                            confidence=0.9,
                            features=SimpleNamespace(
                                motion_score=0.9,
                                interaction_score=0.8,
                                crowd_score=0.1,
                                camera_dynamics_score=0.1,
                            ),
                        ),
                        SimpleNamespace(
                            track_id="trk1",
                            scene_index=1,
                            start_seconds=3.0,
                            end_seconds=4.0,
                            label_hint="person",
                            confidence=0.85,
                            features=SimpleNamespace(
                                motion_score=0.8,
                                interaction_score=0.75,
                                crowd_score=0.15,
                                camera_dynamics_score=0.1,
                            ),
                        ),
                    ]
                )
            ),
            embedding_client=SimpleNamespace(
                embed_images=lambda _image_paths: [
                    SimpleNamespace(track_id="trk0", model_name="dinov2", vector=[1.0, 0.0]),
                    SimpleNamespace(track_id="trk1", model_name="dinov2", vector=[1.0, 0.0]),
                ]
            ),
            label_client=SimpleNamespace(
                score_image_labels=lambda **_kwargs: [
                    SimpleNamespace(label="person", score=0.9),
                    SimpleNamespace(label="object", score=0.1),
                ],
                score_labels=lambda **_kwargs: SimpleNamespace(
                    group_id="cg0",
                    label="person",
                    scores=[
                        SimpleNamespace(label="person", score=0.9),
                        SimpleNamespace(label="object", score=0.1),
                    ],
                ),
            ),
        )

        context = build_tool_context(
            "/tmp/video.mp4",
            options=InspectOptions(),
            tooling_runtime=cast(ToolingRuntime, fake_runtime),
        )

        hints = str(context["tool_grouping_hints"])
        self.assertIn("Embedding/SAM3 grouping hints:", hints)
        self.assertIn("Label hints:", hints)
        self.assertIn("Routing/model-selection hints:", hints)
        self.assertIn("Verify/group hints:", hints)
        self.assertIn("cg0", str(context["tool_routing_hints"]))
        self.assertIn("recommend=VTA", str(context["tool_routing_hints"]))
        self.assertIn("priority=high", str(context["tool_verify_hints"]))
        self.assertIn("cross-scene cluster", str(context["tool_verify_hints"]))


if __name__ == "__main__":
    unittest.main()
