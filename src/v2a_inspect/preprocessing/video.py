from __future__ import annotations

import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

from v2a_inspect.models import VideoAsset

PREPARED_WIDTH = 1280
PREPARED_HEIGHT = 720
PREPARED_FPS = 30


@dataclass(frozen=True)
class PreparedVideoProbe:
    """Operational metadata for a prepared working video."""

    path: Path
    width: int
    height: int
    fps: int
    frame_count: int
    has_audio: bool


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


def probe_prepared_video(video_path: Path) -> PreparedVideoProbe:
    """Probe and validate a prepared working video using PyAV."""

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")

    with av.open(str(video_path)) as container:
        video_streams = [
            stream for stream in container.streams if stream.type == "video"
        ]
        audio_streams = [
            stream for stream in container.streams if stream.type == "audio"
        ]

        if len(video_streams) != 1:
            raise ValueError(
                f"Prepared video must contain exactly one video stream, got {len(video_streams)}"
            )

        video_stream = video_streams[0]
        width = int(getattr(video_stream, "width"))
        height = int(getattr(video_stream, "height"))
        fps = _read_stream_fps(video_stream.average_rate)
        frame_count = sum(1 for _frame in container.decode(video=0))

    probe = PreparedVideoProbe(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        has_audio=bool(audio_streams),
    )
    _validate_prepared_video_probe(probe)
    return probe


def prepare_video(raw_video_path: Path, work_dir: Path) -> VideoAsset:
    """Normalize a raw video and return the prepared pipeline VideoAsset."""

    prepared_path = work_dir / f"{raw_video_path.stem}.prepared.webm"
    normalize_video(raw_video_path, prepared_path)
    probe = probe_prepared_video(prepared_path)
    return VideoAsset(source_path=probe.path, frame_count=probe.frame_count)


def _read_stream_fps(rate: Fraction | None) -> int:
    if rate is None:
        raise ValueError("Prepared video stream is missing average frame rate")

    fps = float(rate)
    if abs(fps - PREPARED_FPS) > 0.001:
        raise ValueError(f"Prepared video must be {PREPARED_FPS}fps, got {fps:g}")
    return PREPARED_FPS


def _validate_prepared_video_probe(probe: PreparedVideoProbe) -> None:
    if probe.width != PREPARED_WIDTH:
        raise ValueError(
            f"Prepared video width must be {PREPARED_WIDTH}, got {probe.width}"
        )
    if probe.height != PREPARED_HEIGHT:
        raise ValueError(
            f"Prepared video height must be {PREPARED_HEIGHT}, got {probe.height}"
        )
    if probe.fps != PREPARED_FPS:
        raise ValueError(f"Prepared video fps must be {PREPARED_FPS}, got {probe.fps}")
    if probe.frame_count <= 0:
        raise ValueError("Prepared video must contain at least one frame")
    if probe.has_audio:
        raise ValueError("Prepared video must not contain audio streams")
