from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image

PREPARED_WIDTH = 1280
PREPARED_HEIGHT = 720
PREPARED_FPS = 30


@dataclass(frozen=True)
class PreparedVideoProbe:
    """Operational metadata for a prepared working video."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool


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
        if video_stream.average_rate is None:
            raise ValueError("Prepared video stream is missing average frame rate")
        fps = float(video_stream.average_rate)
        frame_count = sum(1 for _frame in container.decode(video=0))

    probe = PreparedVideoProbe(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        has_audio=bool(audio_streams),
    )
    validate_prepared_video_probe(probe)
    return probe


def extract_frame(video_path: Path, frame_index: int) -> Image.Image:
    """Extract one decoded video frame by zero-based frame index."""

    if frame_index < 0:
        raise ValueError("frame_index must be greater than or equal to 0")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")

    with av.open(str(video_path)) as container:
        video_streams = [
            stream for stream in container.streams if stream.type == "video"
        ]
        if len(video_streams) != 1:
            raise ValueError(
                f"Video must contain exactly one video stream, got {len(video_streams)}"
            )

        video_stream = video_streams[0]
        if video_stream.time_base is None:
            raise ValueError("Video stream is missing time base")
        if video_stream.average_rate is None:
            raise ValueError("Video stream is missing average frame rate")

        fps = float(video_stream.average_rate)
        target_seconds = frame_index / fps
        target_pts = int(target_seconds / float(video_stream.time_base))
        container.seek(target_pts, stream=video_stream, backward=True)

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            decoded_seconds = float(frame.pts * video_stream.time_base)
            decoded_frame_index = round(decoded_seconds * fps)
            if decoded_frame_index >= frame_index:
                return frame.to_image().convert("RGB")

    raise ValueError(f"Could not decode frame {frame_index} from video: {video_path}")


def validate_prepared_video_probe(probe: PreparedVideoProbe) -> None:
    if probe.width != PREPARED_WIDTH:
        raise ValueError(
            f"Prepared video width must be {PREPARED_WIDTH}, got {probe.width}"
        )
    if probe.height != PREPARED_HEIGHT:
        raise ValueError(
            f"Prepared video height must be {PREPARED_HEIGHT}, got {probe.height}"
        )
    if abs(probe.fps - PREPARED_FPS) > 0.001:
        raise ValueError(
            f"Prepared video fps must be {PREPARED_FPS}, got {probe.fps:g}"
        )
    if probe.frame_count <= 0:
        raise ValueError("Prepared video must contain at least one frame")
    if probe.has_audio:
        raise ValueError("Prepared video must not contain audio streams")
