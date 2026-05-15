from __future__ import annotations

import subprocess
from pathlib import Path

from v2a_inspect.media_utils import (
    PREPARED_FPS,
    PREPARED_HEIGHT,
    PREPARED_WIDTH,
    probe_prepared_video,
)
from v2a_inspect.models import VideoAsset


def normalize_video(raw_video_path: Path, output_path: Path) -> Path:
    """
    Normalize a raw video into the fixed working-video format.

    The output is 1280x720, 30fps, VP9 WebM, yuv420p, and audio-free.
    Non-16:9 inputs are aspect-preserved and padded instead of stretched.
    """

    if not raw_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {raw_video_path}")
    if not raw_video_path.is_file():
        raise ValueError(f"Video path is not a file: {raw_video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_filter = ",".join(
        [
            f"fps={PREPARED_FPS}",
            (
                f"scale={PREPARED_WIDTH}:{PREPARED_HEIGHT}:"
                "force_original_aspect_ratio=decrease"
            ),
            f"pad={PREPARED_WIDTH}:{PREPARED_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
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
        str(raw_video_path),
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        "24",
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

    prepared_path = work_dir / f"{raw_video_path.stem}.prepared.webm"
    normalize_video(raw_video_path, prepared_path)
    probe = probe_prepared_video(prepared_path)
    return VideoAsset(source_path=probe.path, frame_count=probe.frame_count)
