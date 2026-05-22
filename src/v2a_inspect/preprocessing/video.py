from __future__ import annotations

import subprocess
from pathlib import Path

from v2a_inspect.media_utils import (
    PREPARED_FPS,
    PREPARED_HEIGHT,
    PREPARED_WIDTH,
    SAM3_TRACKING_FPS,
    SAM3_TRACKING_HEIGHT,
    SAM3_TRACKING_WIDTH,
    probe_prepared_video,
    probe_sam3_tracking_video,
)
from v2a_inspect.models import VideoAsset


def normalize_video(raw_video_path: Path, output_path: Path) -> Path:
    """
    Normalize a raw video into the fixed working-video format.

    The output is 1280x720, 30fps, H.264 MP4, yuv420p, and audio-free.
    Non-16:9 inputs are aspect-preserved and padded instead of stretched.
    """

    return _normalize_h264_video(
        raw_video_path,
        output_path,
        width=PREPARED_WIDTH,
        height=PREPARED_HEIGHT,
        fps=PREPARED_FPS,
    )


def normalize_sam3_tracking_video(prepared_video_path: Path, output_path: Path) -> Path:
    """Create the lower-resolution, frame-aligned SAM3 tracking video."""

    return _normalize_h264_video(
        prepared_video_path,
        output_path,
        width=SAM3_TRACKING_WIDTH,
        height=SAM3_TRACKING_HEIGHT,
        fps=SAM3_TRACKING_FPS,
    )


def _normalize_h264_video(
    input_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Video path is not a file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_filter = ",".join(
        [
            f"fps={fps}",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
        ]
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-g",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError("ffmpeg executable not found") from exc

    if result.returncode != 0:
        message = result.stderr.strip() or "ffmpeg failed without stderr output"
        raise RuntimeError(f"Could not normalize video: {message}")

    return output_path


def prepare_video(raw_video_path: Path, work_dir: Path) -> VideoAsset:
    """Normalize a raw video and return the prepared pipeline VideoAsset."""

    prepared_path = work_dir / f"{raw_video_path.stem}.prepared.mp4"
    normalize_video(raw_video_path, prepared_path)
    probe = probe_prepared_video(prepared_path)

    sam3_tracking_path = work_dir / f"{raw_video_path.stem}.sam3.mp4"
    normalize_sam3_tracking_video(prepared_path, sam3_tracking_path)
    sam3_probe = probe_sam3_tracking_video(sam3_tracking_path)
    if sam3_probe.frame_count != probe.frame_count:
        raise ValueError(
            "SAM3 tracking video frame count must match prepared video frame count; "
            f"got {sam3_probe.frame_count} and {probe.frame_count}"
        )

    return VideoAsset(
        source_path=probe.path,
        sam3_tracking_path=sam3_probe.path,
        frame_count=probe.frame_count,
    )
