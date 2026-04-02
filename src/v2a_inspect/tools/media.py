from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .types import FrameBatch, SampledFrame, SceneBoundary, VideoProbe


def probe_video(video_path: str) -> VideoProbe:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    format_payload = payload.get("format", {})
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        {},
    )
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    avg_frame_rate = video_stream.get("avg_frame_rate")
    fps = _parse_frame_rate(avg_frame_rate)
    duration = float(format_payload.get("duration") or 0.0)
    frame_count = (
        int(video_stream["nb_frames"])
        if str(video_stream.get("nb_frames", "")).isdigit()
        else None
    )
    return VideoProbe(
        video_path=video_path,
        duration_seconds=duration,
        fps=fps,
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        frame_count=frame_count,
        codec_name=video_stream.get("codec_name"),
        format_name=format_payload.get("format_name"),
        has_audio_stream=has_audio,
    )


def detect_scenes(
    video_path: str,
    *,
    probe: VideoProbe | None = None,
    target_scene_seconds: float = 5.0,
) -> list[SceneBoundary]:
    resolved_probe = probe or probe_video(video_path)
    if resolved_probe.duration_seconds <= 0.0:
        return [SceneBoundary(scene_index=0, start_seconds=0.0, end_seconds=0.0)]

    scenes: list[SceneBoundary] = []
    start = 0.0
    scene_index = 0
    while start < resolved_probe.duration_seconds:
        end = min(start + target_scene_seconds, resolved_probe.duration_seconds)
        scenes.append(
            SceneBoundary(
                scene_index=scene_index,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                strategy="fixed_window",
            )
        )
        if end >= resolved_probe.duration_seconds:
            break
        start = end
        scene_index += 1
    return scenes


def sample_frames(
    video_path: str,
    scenes: list[SceneBoundary],
    *,
    output_dir: str,
    frames_per_scene: int = 3,
) -> list[FrameBatch]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    frame_batches: list[FrameBatch] = []
    for scene in scenes:
        timestamps = _scene_sample_timestamps(scene, frames_per_scene=frames_per_scene)
        frames: list[SampledFrame] = []
        for frame_index, timestamp in enumerate(timestamps):
            frame_path = (
                output_root
                / f"scene_{scene.scene_index:04d}_frame_{frame_index:02d}.jpg"
            )
            _extract_single_frame(video_path, timestamp, frame_path)
            frames.append(
                SampledFrame(
                    scene_index=scene.scene_index,
                    timestamp_seconds=timestamp,
                    image_path=str(frame_path),
                )
            )
        frame_batches.append(FrameBatch(scene_index=scene.scene_index, frames=frames))
    return frame_batches


def _extract_single_frame(
    video_path: str, timestamp_seconds: float, output_path: Path
) -> None:
    primary_command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(
            primary_command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        fallback_command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-ss",
            f"{timestamp_seconds:.3f}",
            "-frames:v",
            "1",
            str(output_path),
        ]
        try:
            subprocess.run(
                fallback_command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            final_fallback_command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-frames:v",
                "1",
                str(output_path),
            ]
            subprocess.run(
                final_fallback_command,
                check=True,
                capture_output=True,
                text=True,
            )


def _scene_sample_timestamps(
    scene: SceneBoundary,
    *,
    frames_per_scene: int,
) -> list[float]:
    if frames_per_scene <= 0:
        return []
    duration = max(scene.end_seconds - scene.start_seconds, 0.0)
    if duration == 0.0:
        return [scene.start_seconds]

    step = duration / (frames_per_scene + 1)
    return [
        round(scene.start_seconds + step * index, 3)
        for index in range(1, frames_per_scene + 1)
    ]


def _parse_frame_rate(raw_rate: str | None) -> float | None:
    if not raw_rate or raw_rate == "0/0":
        return None
    if "/" not in raw_rate:
        return float(raw_rate)
    numerator, denominator = raw_rate.split("/", 1)
    if denominator == "0":
        return None
    return float(numerator) / float(denominator)
