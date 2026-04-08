from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from v2a_inspect.contracts import PhysicalSourceTrack, TrackCrop
from v2a_inspect_server.runtime import ToolingRuntime
from v2a_inspect_server.tool_context import build_tool_context


class CropEvidenceFlowTests(unittest.TestCase):
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
    def test_embeddings_and_labels_use_crop_paths(
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
        mock_probe_video.return_value = SimpleNamespace(
            duration_seconds=3.0, fps=2.0, width=320, height=240
        )
        mock_build_candidate_cuts.return_value = [
            SimpleNamespace(cut_id="cut-0000", timestamp_seconds=1.0)
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
                    SimpleNamespace(image_path="/tmp/frame0.jpg", timestamp_seconds=0.0)
                ],
            )
        ]
        mock_generate_storyboard.return_value = "/tmp/storyboard.jpg"
        mock_export_window_clips.return_value = {}
        mock_hydrate_evidence_windows.return_value = [
            SimpleNamespace(
                window_id="window-0000",
                start_time=0.0,
                end_time=3.0,
                cut_refs=["cut-0000"],
                artifact_refs=["/tmp/storyboard.jpg"],
                sampled_frame_ids=["/tmp/frame0.jpg"],
            )
        ]
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
        mock_build_provisional_source_tracks.return_value = [
            PhysicalSourceTrack(
                source_id="source-0000",
                kind="foreground",
                label_candidates=[],
                spans=[(0.0, 1.0)],
                evidence_refs=["trk0-crop-00"],
                identity_confidence=0.9,
                reid_neighbors=[],
                temporary_adapter_from="Sam3EntityTrack",
            )
        ]

        embedding_client = SimpleNamespace(
            embed_images=MagicMock(
                return_value=[
                    SimpleNamespace(
                        track_id="trk0", model_name="dinov2", vector=[1.0, 0.0]
                    )
                ]
            )
        )
        label_client = SimpleNamespace(
            score_image_labels=MagicMock(
                return_value=[SimpleNamespace(label="person", score=0.9)]
            ),
            score_labels=MagicMock(
                return_value=SimpleNamespace(
                    group_id="cg0",
                    label="person",
                    scores=[SimpleNamespace(label="person", score=0.9)],
                )
            ),
        )
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
                            points=[
                                SimpleNamespace(
                                    timestamp_seconds=0.0,
                                    bbox_xyxy=[0, 0, 10, 10],
                                    mask_rle=None,
                                )
                            ],
                        )
                    ]
                )
            ),
            embedding_client=embedding_client,
            label_client=label_client,
        )

        build_tool_context(
            "/tmp/video.mp4",
            options=SimpleNamespace(enable_vlm_verify=True),
            tooling_runtime=cast(ToolingRuntime, fake_runtime),
        )

        self.assertEqual(
            embedding_client.embed_images.call_args.args[0],
            {"trk0": ["/tmp/crop0.jpg"]},
        )
        crop_label_calls = [
            call.kwargs["image_paths"]
            for call in label_client.score_image_labels.call_args_list
        ]
        self.assertEqual(crop_label_calls, [["/tmp/crop0.jpg"]])


if __name__ == "__main__":
    unittest.main()
