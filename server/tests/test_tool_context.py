from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from v2a_inspect.contracts import LabelCandidate, PhysicalSourceTrack, TrackCrop
from v2a_inspect.workflows import InspectOptions
from v2a_inspect_server.runtime import ToolingRuntime
from v2a_inspect_server.tool_context import build_tool_context


class ToolContextTests(unittest.TestCase):
    def _structure_setup(
        self,
        *,
        mock_probe_video,
        mock_build_candidate_cuts,
        mock_build_evidence_windows,
        mock_scene_boundaries,
        mock_frame_output_dir,
        mock_sample_frames,
        mock_generate_storyboard,
        mock_export_window_clips,
        mock_hydrate_evidence_windows,
        duration_seconds: float,
        window_count: int,
    ) -> None:
        mock_probe_video.return_value = SimpleNamespace(
            duration_seconds=duration_seconds,
            fps=2.0,
            width=320,
            height=240,
        )
        mock_build_candidate_cuts.return_value = [
            SimpleNamespace(cut_id=f"cut-{index:04d}", timestamp_seconds=float(index + 1))
            for index in range(window_count)
        ]
        mock_build_evidence_windows.return_value = [
            SimpleNamespace(
                window_id=f"window-{index:04d}",
                start_time=float(index * 3),
                end_time=float((index + 1) * 3),
                cut_refs=[f"cut-{index:04d}"],
                artifact_refs=[],
            )
            for index in range(window_count)
        ]
        mock_scene_boundaries.return_value = [
            SimpleNamespace(
                scene_index=index,
                start_seconds=float(index * 3),
                end_seconds=float((index + 1) * 3),
                strategy="ffmpeg_scene_detect",
            )
            for index in range(window_count)
        ]
        mock_frame_output_dir.return_value = Path("/tmp/frame-root")
        mock_sample_frames.return_value = [
            SimpleNamespace(
                scene_index=index,
                frames=[
                    SimpleNamespace(
                        image_path=f"/tmp/frame{index}.jpg",
                        timestamp_seconds=float(index),
                    )
                ],
            )
            for index in range(window_count)
        ]
        mock_generate_storyboard.return_value = "/tmp/storyboard.jpg"
        mock_export_window_clips.return_value = {}
        mock_hydrate_evidence_windows.return_value = [
            SimpleNamespace(
                window_id=f"window-{index:04d}",
                start_time=float(index * 3),
                end_time=float((index + 1) * 3),
                cut_refs=[f"cut-{index:04d}"],
                artifact_refs=["/tmp/storyboard.jpg"],
                sampled_frame_ids=[f"/tmp/frame{index}.jpg"],
            )
            for index in range(window_count)
        ]

    @patch("v2a_inspect_server.tool_context.build_provisional_source_tracks")
    @patch("v2a_inspect_server.tool_context.build_identity_edges")
    @patch("v2a_inspect_server.tool_context.group_crop_paths_by_track")
    @patch("v2a_inspect_server.tool_context.crop_tracks")
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
        mock_crop_tracks,
        mock_group_crop_paths_by_track,
        mock_build_identity_edges,
        mock_build_provisional_source_tracks,
    ) -> None:
        self._structure_setup(
            mock_probe_video=mock_probe_video,
            mock_build_candidate_cuts=mock_build_candidate_cuts,
            mock_build_evidence_windows=mock_build_evidence_windows,
            mock_scene_boundaries=mock_scene_boundaries,
            mock_frame_output_dir=mock_frame_output_dir,
            mock_sample_frames=mock_sample_frames,
            mock_generate_storyboard=mock_generate_storyboard,
            mock_export_window_clips=mock_export_window_clips,
            mock_hydrate_evidence_windows=mock_hydrate_evidence_windows,
            duration_seconds=3.0,
            window_count=1,
        )
        mock_crop_tracks.return_value = []
        mock_group_crop_paths_by_track.return_value = {}
        mock_build_identity_edges.return_value = []
        mock_build_provisional_source_tracks.return_value = []

        context = build_tool_context("/tmp/video.mp4", options=InspectOptions())

        self.assertIn("candidate_cuts", context)
        self.assertIn("evidence_windows", context)
        self.assertIn("storyboard_path", context)
        self.assertIn("Candidate cuts:", str(context["tool_scene_summary"]))
        self.assertIn("Evidence windows:", str(context["tool_scene_summary"]))
        self.assertEqual(len(context["progress_messages"]), 5)

    @patch("v2a_inspect_server.tool_context.build_provisional_source_tracks")
    @patch("v2a_inspect_server.tool_context.build_identity_edges")
    @patch("v2a_inspect_server.tool_context.group_crop_paths_by_track")
    @patch("v2a_inspect_server.tool_context.crop_tracks")
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
        mock_crop_tracks,
        mock_group_crop_paths_by_track,
        mock_build_identity_edges,
        mock_build_provisional_source_tracks,
    ) -> None:
        self._structure_setup(
            mock_probe_video=mock_probe_video,
            mock_build_candidate_cuts=mock_build_candidate_cuts,
            mock_build_evidence_windows=mock_build_evidence_windows,
            mock_scene_boundaries=mock_scene_boundaries,
            mock_frame_output_dir=mock_frame_output_dir,
            mock_sample_frames=mock_sample_frames,
            mock_generate_storyboard=mock_generate_storyboard,
            mock_export_window_clips=mock_export_window_clips,
            mock_hydrate_evidence_windows=mock_hydrate_evidence_windows,
            duration_seconds=3.0,
            window_count=1,
        )
        mock_crop_tracks.return_value = [
            TrackCrop(
                crop_id="trk0-crop-00",
                track_id="trk0",
                scene_index=0,
                frame_path="/tmp/frame0.jpg",
                crop_path="/tmp/crop0.jpg",
                timestamp_seconds=0.0,
                bbox_xyxy=[0, 0, 10, 10],
            )
        ]
        mock_group_crop_paths_by_track.return_value = {"trk0": ["/tmp/crop0.jpg"]}
        mock_build_identity_edges.return_value = []
        mock_build_provisional_source_tracks.return_value = []
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

        self.assertIn("track_crops", context)
        self.assertIn("tool_grouping_hints", context)
        self.assertIn("tool_routing_hints", context)
        self.assertIn("tool_verify_hints", context)
        self.assertIn("routing_decisions", context)
        self.assertIn("SAM3 track hints:", str(context["tool_grouping_hints"]))
        self.assertIn("Routing/model-selection hints:", str(context["tool_routing_hints"]))
        self.assertIn("track:trk0", str(context["tool_routing_hints"]))
        self.assertIn("Verify/group hints:", str(context["tool_verify_hints"]))
        self.assertIn("priority=low", str(context["tool_verify_hints"]))

    @patch("v2a_inspect_server.tool_context.build_provisional_source_tracks")
    @patch("v2a_inspect_server.tool_context.build_identity_edges")
    @patch("v2a_inspect_server.tool_context.group_crop_paths_by_track")
    @patch("v2a_inspect_server.tool_context.crop_tracks")
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
        mock_crop_tracks,
        mock_group_crop_paths_by_track,
        mock_build_identity_edges,
        mock_build_provisional_source_tracks,
    ) -> None:
        self._structure_setup(
            mock_probe_video=mock_probe_video,
            mock_build_candidate_cuts=mock_build_candidate_cuts,
            mock_build_evidence_windows=mock_build_evidence_windows,
            mock_scene_boundaries=mock_scene_boundaries,
            mock_frame_output_dir=mock_frame_output_dir,
            mock_sample_frames=mock_sample_frames,
            mock_generate_storyboard=mock_generate_storyboard,
            mock_export_window_clips=mock_export_window_clips,
            mock_hydrate_evidence_windows=mock_hydrate_evidence_windows,
            duration_seconds=6.0,
            window_count=2,
        )
        mock_crop_tracks.return_value = [
            TrackCrop(crop_id="trk0-crop-00", track_id="trk0", scene_index=0, frame_path="/tmp/frame0.jpg", crop_path="/tmp/crop0.jpg", timestamp_seconds=0.0, bbox_xyxy=[0, 0, 10, 10]),
            TrackCrop(crop_id="trk1-crop-00", track_id="trk1", scene_index=1, frame_path="/tmp/frame1.jpg", crop_path="/tmp/crop1.jpg", timestamp_seconds=3.0, bbox_xyxy=[0, 0, 10, 10]),
        ]
        mock_group_crop_paths_by_track.return_value = {"trk0": ["/tmp/crop0.jpg"], "trk1": ["/tmp/crop1.jpg"]}
        mock_build_identity_edges.return_value = [
            SimpleNamespace(source_track_id="trk0", target_track_id="trk1", confidence=0.95, same_window=False, accepted=True)
        ]
        mock_build_provisional_source_tracks.return_value = [
            PhysicalSourceTrack(
                source_id="source-0000",
                kind="foreground",
                label_candidates=[LabelCandidate(label="person", score=0.9)],
                spans=[(0.0, 1.0), (3.0, 4.0)],
                crop_refs=["trk0-crop-00", "trk1-crop-00"],
                track_refs=["trk0", "trk1"],
                identity_confidence=0.95,
                reid_neighbors=[],
                temporary_adapter_from="Sam3EntityTrack",
            )
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
                            features=SimpleNamespace(motion_score=0.9, interaction_score=0.8, crowd_score=0.1, camera_dynamics_score=0.1),
                        ),
                        SimpleNamespace(
                            track_id="trk1",
                            scene_index=1,
                            start_seconds=3.0,
                            end_seconds=4.0,
                            label_hint="person",
                            confidence=0.85,
                            features=SimpleNamespace(motion_score=0.8, interaction_score=0.75, crowd_score=0.15, camera_dynamics_score=0.1),
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
        self.assertIn("Source identity hints:", hints)
        self.assertIn("track_crops", context)
        self.assertIn("identity_edges", context)
        self.assertIn("physical_sources", context)

    @patch("v2a_inspect_server.tool_context.build_provisional_source_tracks")
    @patch("v2a_inspect_server.tool_context.build_identity_edges")
    @patch("v2a_inspect_server.tool_context.group_crop_paths_by_track")
    @patch("v2a_inspect_server.tool_context.crop_tracks")
    @patch("v2a_inspect_server.tool_context.hydrate_evidence_windows")
    @patch("v2a_inspect_server.tool_context.export_window_clips")
    @patch("v2a_inspect_server.tool_context.generate_storyboard")
    @patch("v2a_inspect_server.tool_context.sample_frames")
    @patch("v2a_inspect_server.tool_context._frame_output_dir")
    @patch("v2a_inspect_server.tool_context.evidence_windows_to_scene_boundaries")
    @patch("v2a_inspect_server.tool_context.build_evidence_windows")
    @patch("v2a_inspect_server.tool_context.build_candidate_cuts")
    @patch("v2a_inspect_server.tool_context.probe_video")
    def test_build_tool_context_attributes_embedding_failures_separately(
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
        mock_crop_tracks,
        mock_group_crop_paths_by_track,
        mock_build_identity_edges,
        mock_build_provisional_source_tracks,
    ) -> None:
        self._structure_setup(
            mock_probe_video=mock_probe_video,
            mock_build_candidate_cuts=mock_build_candidate_cuts,
            mock_build_evidence_windows=mock_build_evidence_windows,
            mock_scene_boundaries=mock_scene_boundaries,
            mock_frame_output_dir=mock_frame_output_dir,
            mock_sample_frames=mock_sample_frames,
            mock_generate_storyboard=mock_generate_storyboard,
            mock_export_window_clips=mock_export_window_clips,
            mock_hydrate_evidence_windows=mock_hydrate_evidence_windows,
            duration_seconds=3.0,
            window_count=1,
        )
        mock_crop_tracks.return_value = [
            TrackCrop(
                crop_id="trk0-crop-00",
                track_id="trk0",
                scene_index=0,
                frame_path="/tmp/frame0.jpg",
                crop_path="/tmp/crop0.jpg",
                timestamp_seconds=0.0,
                bbox_xyxy=[0, 0, 10, 10],
            )
        ]
        mock_group_crop_paths_by_track.return_value = {"trk0": ["/tmp/crop0.jpg"]}
        mock_build_identity_edges.return_value = []
        mock_build_provisional_source_tracks.return_value = []
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
                            points=[],
                        )
                    ]
                )
            ),
            embedding_client=SimpleNamespace(
                embed_images=lambda _track_image_paths: (_ for _ in ()).throw(
                    RuntimeError("embedding failed")
                )
            ),
            label_client=SimpleNamespace(
                score_image_labels=lambda **_kwargs: [],
                score_labels=lambda **_kwargs: SimpleNamespace(
                    group_id="g0",
                    label="object",
                    scores=[],
                ),
            ),
        )

        context = build_tool_context(
            "/tmp/video.mp4",
            options=InspectOptions(),
            tooling_runtime=cast(ToolingRuntime, fake_runtime),
        )

        warnings = cast(list[str], context.get("warnings", []))
        self.assertEqual(len(warnings), 1)
        self.assertIn("Crop embedding/label enrichment unavailable", warnings[0])


if __name__ == "__main__":
    unittest.main()
