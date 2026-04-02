from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from v2a_inspect.settings import settings
from v2a_inspect.tools import detect_scenes, probe_video, sample_frames
from v2a_inspect.workflows import InspectOptions


def build_tool_context(
    video_path: str,
    *,
    options: InspectOptions,
) -> dict[str, object]:
    probe = probe_video(video_path)
    scenes = detect_scenes(video_path, probe=probe, target_scene_seconds=5.0)
    frame_root = _frame_output_dir(video_path)
    frame_batches = sample_frames(
        video_path,
        scenes,
        output_dir=str(frame_root),
        frames_per_scene=2,
    )

    return {
        "video_probe": probe,
        "scene_boundaries": scenes,
        "frame_batches": frame_batches,
        "tool_scene_summary": _scene_summary(probe=probe, scenes=scenes),
        "progress_messages": [
            f"Tool pipeline: probed video ({probe.width}x{probe.height}, {probe.duration_seconds:.1f}s).",
            f"Tool pipeline: planned {len(scenes)} scene windows.",
            f"Tool pipeline: sampled {sum(len(batch.frames) for batch in frame_batches)} frames.",
        ],
    }


def _frame_output_dir(video_path: str) -> Path:
    base_dir = settings.shared_video_dir or Path(tempfile.gettempdir())
    resolved_base = Path(base_dir)
    try:
        resolved_base.mkdir(parents=True, exist_ok=True)
    except OSError:
        resolved_base = Path(tempfile.gettempdir())
        resolved_base.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f"{Path(video_path).stem}_frames_",
            dir=str(resolved_base),
        )
    )


def _scene_summary(*, probe: object, scenes: Sequence[object]) -> str:
    probe_duration = getattr(probe, "duration_seconds", None)
    probe_fps = getattr(probe, "fps", None)
    probe_width = getattr(probe, "width", None)
    probe_height = getattr(probe, "height", None)
    lines = [
        f"- scene {getattr(scene, 'scene_index')}: {getattr(scene, 'start_seconds'):.1f}s to {getattr(scene, 'end_seconds'):.1f}s ({getattr(scene, 'strategy')})"
        for scene in scenes[:12]
    ]
    return "\n".join(
        [
            f"Probe: duration={probe_duration:.1f}s, fps={probe_fps}, resolution={probe_width}x{probe_height}.",
            "Tool-detected scene windows:",
            *lines,
        ]
    )
