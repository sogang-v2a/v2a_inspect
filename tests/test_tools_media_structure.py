from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from v2a_inspect.contracts import CandidateCut, CutReason
from v2a_inspect.tools.media import (
    build_candidate_cuts,
    build_evidence_windows,
    export_window_clips,
    generate_storyboard,
    merge_candidate_cuts,
    probe_video,
    sample_frames,
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
        self.assertTrue(any(reason.kind == "shot_boundary" for cut in cuts for reason in cut.reasons))
        self.assertTrue(any(0.5 <= cut.timestamp_seconds <= 1.5 for cut in cuts))

    def test_merge_candidate_cuts_deduplicates_close_timestamps(self) -> None:
        cuts = merge_candidate_cuts(
            [
                CandidateCut(
                    cut_id="a",
                    timestamp_seconds=1.0,
                    confidence=0.8,
                    reasons=[CutReason(kind="shot_boundary", confidence=0.8, rationale="a")],
                ),
                CandidateCut(
                    cut_id="b",
                    timestamp_seconds=1.2,
                    confidence=0.4,
                    reasons=[CutReason(kind="composition_change", confidence=0.4, rationale="b")],
                ),
            ],
            minimum_spacing_seconds=0.5,
        )
        self.assertEqual(len(cuts), 1)
        self.assertEqual({reason.kind for reason in cuts[0].reasons}, {"shot_boundary", "composition_change"})

    def test_build_evidence_windows_respects_minimum_window_length(self) -> None:
        fake_probe = type("Probe", (), {"duration_seconds": 4.0})()
        windows = build_evidence_windows(
            probe=fake_probe,
            candidate_cuts=[
                CandidateCut(cut_id="c0", timestamp_seconds=0.2, confidence=0.2, reasons=[]),
                CandidateCut(cut_id="c1", timestamp_seconds=1.5, confidence=0.9, reasons=[]),
                CandidateCut(cut_id="c2", timestamp_seconds=3.8, confidence=0.9, reasons=[]),
            ],
            minimum_window_seconds=1.0,
        )
        self.assertTrue(all((window.end_time - window.start_time) >= 1.0 for window in windows[:-1]))

    def test_storyboard_and_optional_clip_exports_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            video_path = root / "three_scene.mp4"
            _build_three_scene_clip(video_path)
            probe = probe_video(str(video_path))
            cuts = build_candidate_cuts(str(video_path), probe=probe, target_scene_seconds=10.0)
            windows = build_evidence_windows(probe=probe, candidate_cuts=cuts)
            scenes = [
                type("Scene", (), {"scene_index": idx, "start_seconds": window.start_time, "end_seconds": window.end_time, "strategy": "fixed_window"})()
                for idx, window in enumerate(windows)
            ]
            frame_batches = sample_frames(str(video_path), scenes, output_dir=str(root / "frames"), frames_per_scene=1)
            storyboard_path = generate_storyboard(frame_batches, output_path=str(root / "storyboard.jpg"))
            clip_paths = export_window_clips(str(video_path), windows, output_dir=str(root / "clips"), max_windows=1)
            self.assertTrue(Path(storyboard_path).exists())
            self.assertEqual(len(clip_paths), 1)
            self.assertTrue(Path(next(iter(clip_paths.values()))).exists())


if __name__ == "__main__":
    unittest.main()
