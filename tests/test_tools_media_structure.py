from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2a_inspect.contracts import CandidateCut, CutReason, LabelCandidate
from v2a_inspect.tools.media import (
    _media_binary,
    _probe_video_with_ffmpeg,
    build_candidate_cuts,
    build_context_candidate_cuts,
    build_evidence_windows,
    export_window_clips,
    generate_storyboard,
    merge_candidate_cuts,
    probe_video,
    sample_frames,
)
from v2a_inspect.tools.types import (
    FrameBatch,
    Sam3EntityTrack,
    Sam3TrackPoint,
    Sam3VisualFeatures,
    SampledFrame,
)


def _build_three_scene_clip(output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=96x96:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=96x96:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=96x96:d=1",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


class MediaStructureTests(unittest.TestCase):
    def test_media_binary_prefers_active_python_environment(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "shutil.which", return_value=None
        ), patch("sys.executable", "/opt/venv/bin/python"):
            with patch.object(Path, "exists", return_value=True):
                self.assertEqual(_media_binary("ffprobe"), "/opt/venv/bin/ffprobe")

    def test_media_binary_uses_imageio_ffmpeg_when_system_binary_is_missing(self) -> None:
        fake_module = SimpleNamespace(get_ffmpeg_exe=lambda: "/tmp/imageio-ffmpeg")
        with patch.dict("os.environ", {}, clear=True), patch(
            "shutil.which", return_value=None
        ), patch.dict("sys.modules", {"imageio_ffmpeg": fake_module}):
            self.assertEqual(_media_binary("ffmpeg"), "/tmp/imageio-ffmpeg")

    def test_probe_video_falls_back_to_ffmpeg_when_ffprobe_is_missing(self) -> None:
        stderr = (
            "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/tmp/video.mp4':\n"
            "  Duration: 00:00:01.00, start: 0.000000, bitrate: 12 kb/s\n"
            "  Stream #0:0: Video: h264, yuv420p(progressive), 96x96, 2 fps, 2 tbr, 16384 tbn\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(stderr=stderr)
            probe = _probe_video_with_ffmpeg("/tmp/video.mp4")
        self.assertEqual(probe.duration_seconds, 1.0)
        self.assertEqual(probe.width, 96)
        self.assertEqual(probe.height, 96)
        self.assertEqual(probe.fps, 2.0)
        self.assertEqual(probe.codec_name, "h264")

    def test_build_candidate_cuts_emits_hard_cut_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "three_scene.mp4"
            _build_three_scene_clip(video_path)
            probe = probe_video(str(video_path))
            cuts = build_candidate_cuts(
                str(video_path),
                probe=probe,
                target_scene_seconds=10.0,
            )
        self.assertTrue(
            any(
                reason.kind == "shot_boundary" for cut in cuts for reason in cut.reasons
            )
        )
        self.assertTrue(any(0.5 <= cut.timestamp_seconds <= 1.5 for cut in cuts))

    def test_merge_candidate_cuts_deduplicates_close_timestamps(self) -> None:
        cuts = merge_candidate_cuts(
            [
                CandidateCut(
                    cut_id="a",
                    timestamp_seconds=1.0,
                    confidence=0.8,
                    reasons=[
                        CutReason(kind="shot_boundary", confidence=0.8, rationale="a")
                    ],
                ),
                CandidateCut(
                    cut_id="b",
                    timestamp_seconds=1.2,
                    confidence=0.4,
                    reasons=[
                        CutReason(
                            kind="composition_change", confidence=0.4, rationale="b"
                        )
                    ],
                ),
            ],
            minimum_spacing_seconds=0.5,
        )
        self.assertEqual(len(cuts), 1)
        self.assertEqual(
            {reason.kind for reason in cuts[0].reasons},
            {"shot_boundary", "composition_change"},
        )

    def test_build_evidence_windows_respects_minimum_window_length(self) -> None:
        fake_probe = type("Probe", (), {"duration_seconds": 4.0})()
        windows = build_evidence_windows(
            probe=fake_probe,
            candidate_cuts=[
                CandidateCut(
                    cut_id="c0", timestamp_seconds=0.2, confidence=0.2, reasons=[]
                ),
                CandidateCut(
                    cut_id="c1", timestamp_seconds=1.5, confidence=0.9, reasons=[]
                ),
                CandidateCut(
                    cut_id="c2", timestamp_seconds=3.8, confidence=0.9, reasons=[]
                ),
            ],
            minimum_window_seconds=1.0,
        )
        self.assertTrue(
            all((window.end_time - window.start_time) >= 1.0 for window in windows[:-1])
        )

    def test_storyboard_and_optional_clip_exports_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            video_path = root / "three_scene.mp4"
            _build_three_scene_clip(video_path)
            probe = probe_video(str(video_path))
            cuts = build_candidate_cuts(
                str(video_path), probe=probe, target_scene_seconds=10.0
            )
            windows = build_evidence_windows(probe=probe, candidate_cuts=cuts)
            scenes = [
                type(
                    "Scene",
                    (),
                    {
                        "scene_index": idx,
                        "start_seconds": window.start_time,
                        "end_seconds": window.end_time,
                        "strategy": "fixed_window",
                    },
                )()
                for idx, window in enumerate(windows)
            ]
            frame_batches = sample_frames(
                str(video_path),
                scenes,
                output_dir=str(root / "frames"),
                frames_per_scene=1,
            )
            storyboard_path = generate_storyboard(
                frame_batches, output_path=str(root / "storyboard.jpg")
            )
            clip_paths = export_window_clips(
                str(video_path), windows, output_dir=str(root / "clips"), max_windows=1
            )
            self.assertTrue(Path(storyboard_path).exists())
            self.assertEqual(len(clip_paths), 1)
            self.assertTrue(Path(next(iter(clip_paths.values()))).exists())

    def test_context_candidate_cuts_preserves_existing_structural_cuts(self) -> None:
        candidate_cuts = [
            CandidateCut(
                cut_id="cut-0000",
                timestamp_seconds=2.0,
                confidence=0.2,
                reasons=[
                    CutReason(
                        kind="composition_change",
                        confidence=0.2,
                        rationale="frame difference",
                    )
                ],
            )
        ]
        probe = type("Probe", (), {"duration_seconds": 6.0})()
        frame_batches = [
            FrameBatch(
                scene_index=0,
                frames=[
                    SampledFrame(
                        scene_index=0,
                        timestamp_seconds=0.5,
                        image_path="/tmp/scene0.jpg",
                    )
                ],
            ),
            FrameBatch(
                scene_index=1,
                frames=[
                    SampledFrame(
                        scene_index=1,
                        timestamp_seconds=3.0,
                        image_path="/tmp/scene1.jpg",
                    )
                ],
            ),
        ]
        tracks = [
            Sam3EntityTrack(
                track_id="trk0",
                scene_index=0,
                start_seconds=0.5,
                end_seconds=1.4,
                confidence=0.8,
                label_hint="cat",
                points=[
                    Sam3TrackPoint(
                        timestamp_seconds=0.5,
                        frame_path="/tmp/scene0.jpg",
                        confidence=0.8,
                        bbox_xyxy=[0, 0, 10, 10],
                    )
                ],
                features=Sam3VisualFeatures(interaction_score=0.7),
            ),
            Sam3EntityTrack(
                track_id="trk1",
                scene_index=1,
                start_seconds=3.0,
                end_seconds=4.0,
                confidence=0.8,
                label_hint="car",
                points=[
                    Sam3TrackPoint(
                        timestamp_seconds=3.0,
                        frame_path="/tmp/scene1.jpg",
                        confidence=0.8,
                        bbox_xyxy=[0, 0, 10, 10],
                    )
                ],
                features=Sam3VisualFeatures(),
            ),
        ]
        label_candidates = {
            "trk0": [LabelCandidate(label="cat", score=0.9)],
            "trk1": [LabelCandidate(label="car", score=0.9)],
        }

        merged_cuts, evidence_windows = build_context_candidate_cuts(
            candidate_cuts=candidate_cuts,
            probe=probe,
            frame_batches=frame_batches,
            tracks=tracks,
            label_candidates_by_track=label_candidates,
            storyboard_path="/tmp/storyboard.jpg",
        )

        self.assertEqual(merged_cuts, candidate_cuts)
        self.assertTrue(any("/tmp/scene0.jpg" in window.sampled_frame_ids for window in evidence_windows))


if __name__ == "__main__":
    unittest.main()
