from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from v2a_inspect.settings import settings
from v2a_inspect.tools import FrameBatch, detect_scenes, probe_video, sample_frames
from v2a_inspect.workflows import InspectOptions

if TYPE_CHECKING:
    from v2a_inspect_server.runtime import ToolingRuntime


def build_tool_context(
    video_path: str,
    *,
    options: InspectOptions,
    tooling_runtime: ToolingRuntime | None = None,
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

    context: dict[str, object] = {
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
    if tooling_runtime is not None:
        _append_runtime_tool_evidence(
            context,
            frame_batches=frame_batches,
            tooling_runtime=tooling_runtime,
        )
    return context


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


def _append_runtime_tool_evidence(
    context: dict[str, object],
    *,
    frame_batches: Sequence[FrameBatch],
    tooling_runtime: ToolingRuntime,
) -> None:
    raw_progress = context.get("progress_messages", [])
    progress_messages = (
        [str(item) for item in raw_progress] if isinstance(raw_progress, list) else []
    )
    try:
        sam3_track_set = tooling_runtime.sam3_client.extract_entities(
            list(frame_batches)
        )
        context["sam3_track_set"] = sam3_track_set
        context["tool_grouping_hints"] = _grouping_hints_from_tracks(sam3_track_set)
        progress_messages.append(
            f"Tool pipeline: extracted {len(sam3_track_set.tracks)} SAM3 tracks."
        )
    except Exception as exc:  # noqa: BLE001
        raw_warnings = context.get("warnings", [])
        warnings = (
            [str(item) for item in raw_warnings]
            if isinstance(raw_warnings, list)
            else []
        )
        warnings.append(f"Tool runtime: SAM3 extraction unavailable ({exc}).")
        context["warnings"] = warnings
    context["progress_messages"] = progress_messages


def _grouping_hints_from_tracks(track_set: object) -> str:
    tracks = getattr(track_set, "tracks", [])
    if not tracks:
        return "SAM3 found no tracks."
    lines = []
    for track in tracks[:20]:
        lines.append(
            f"- {getattr(track, 'track_id')}: scene={getattr(track, 'scene_index')} "
            f"range={getattr(track, 'start_seconds'):.1f}-{getattr(track, 'end_seconds'):.1f} "
            f"label_hint={getattr(track, 'label_hint', None)} confidence={getattr(track, 'confidence', 0.0):.2f}"
        )
    return "\n".join(["SAM3 track hints:", *lines])
