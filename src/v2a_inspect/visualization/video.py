from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
from PIL import Image


def iter_video_frames(
    video_path: Path,
    *,
    start_frame_index: int,
    end_frame_index: int,
    fps: int = 30,
) -> Iterator[tuple[int, Image.Image]]:
    if start_frame_index < 0:
        raise ValueError("start_frame_index must be greater than or equal to 0")
    if end_frame_index <= start_frame_index:
        raise ValueError("end_frame_index must be greater than start_frame_index")

    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        if video_stream.time_base is None:
            raise ValueError("Video stream is missing time base")

        start_seconds = start_frame_index / fps
        start_pts = int(start_seconds / float(video_stream.time_base))
        container.seek(start_pts, stream=video_stream, backward=True)

        yielded_frame_indexes: set[int] = set()
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            seconds = float(frame.pts * video_stream.time_base)
            frame_index = round(seconds * fps)
            if frame_index < start_frame_index:
                continue
            if frame_index >= end_frame_index:
                break
            if frame_index in yielded_frame_indexes:
                continue

            yielded_frame_indexes.add(frame_index)
            yield frame_index, frame.to_image().convert("RGB")


def write_video_frames(
    frames: list[Image.Image],
    output_path: Path,
    *,
    fps: int = 30,
) -> Path:
    if not frames:
        raise ValueError("frames must contain at least one image")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frames[0]
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = first_frame.width
        stream.height = first_frame.height
        stream.pix_fmt = "yuv420p"

        for image in frames:
            video_frame = av.VideoFrame.from_image(image.convert("RGB"))
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return output_path
