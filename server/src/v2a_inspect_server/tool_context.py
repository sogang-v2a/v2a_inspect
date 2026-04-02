from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from v2a_inspect.settings import settings
from v2a_inspect.tools import (
    FrameBatch,
    Sam3EntityTrack,
    group_entity_embeddings,
    detect_scenes,
    probe_video,
    sample_frames,
)
from v2a_inspect.workflows import InspectOptions

if TYPE_CHECKING:
    from v2a_inspect_server.runtime import ToolingRuntime


class _LabelClientLike(Protocol):
    def score_labels(
        self,
        *,
        group_id: str,
        image_paths: list[str],
        labels: list[str],
    ) -> "_LabelResultLike": ...


class _LabelScoreLike(Protocol):
    label: str
    score: float


class _LabelResultLike(Protocol):
    group_id: str
    label: str
    scores: Sequence[_LabelScoreLike]


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
        track_image_paths = _track_image_paths(
            frame_batches=frame_batches,
            tracks=sam3_track_set.tracks,
        )
        if track_image_paths:
            embedding_client = getattr(tooling_runtime, "embedding_client", None)
            label_client = getattr(tooling_runtime, "label_client", None)
            if embedding_client is None or label_client is None:
                context["progress_messages"] = progress_messages
                return

            embeddings = embedding_client.embed_images(track_image_paths)
            context["entity_embeddings"] = embeddings
            tracks_by_id = {track.track_id: track for track in sam3_track_set.tracks}
            candidate_groups = group_entity_embeddings(
                embeddings,
                tracks_by_id=tracks_by_id,
            )
            context["candidate_groups"] = candidate_groups.groups
            context["tool_grouping_hints"] = _grouping_hints_from_groups(
                candidate_groups.groups,
                tracks_by_id=tracks_by_id,
            )
            progress_messages.append(
                f"Tool pipeline: proposed {len(candidate_groups.groups)} embedding-based groups."
            )
            _append_group_label_hints(
                context,
                label_client=label_client,
                candidate_groups=candidate_groups.groups,
                track_image_paths=track_image_paths,
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


def _track_image_paths(
    *,
    frame_batches: Sequence[FrameBatch],
    tracks: Sequence[Sam3EntityTrack],
) -> dict[str, list[str]]:
    images_by_scene = {
        batch.scene_index: [frame.image_path for frame in batch.frames]
        for batch in frame_batches
    }
    track_paths: dict[str, list[str]] = {}
    for track in tracks:
        image_paths = images_by_scene.get(track.scene_index, [])
        if image_paths:
            track_paths[track.track_id] = image_paths
    return track_paths


def _grouping_hints_from_groups(
    groups: Sequence[object],
    *,
    tracks_by_id: dict[str, Sam3EntityTrack],
) -> str:
    if not groups:
        return "Embedding grouping found no candidate groups."
    lines = []
    for group in groups:
        member_ids = list(getattr(group, "member_track_ids", []))
        member_summaries = []
        for member_id in member_ids:
            track = tracks_by_id.get(member_id)
            if track is None:
                continue
            member_summaries.append(
                f"{member_id}(scene={track.scene_index}, conf={track.confidence:.2f}, hint={track.label_hint})"
            )
        lines.append(
            f"- {getattr(group, 'group_id')}: members={', '.join(member_summaries)} confidence={getattr(group, 'confidence', 0.0):.2f}"
        )
    return "\n".join(["Embedding/SAM3 grouping hints:", *lines])


def _append_group_label_hints(
    context: dict[str, object],
    *,
    label_client: _LabelClientLike,
    candidate_groups: Sequence[object],
    track_image_paths: dict[str, list[str]],
) -> None:
    label_lines: list[str] = []
    for group in candidate_groups:
        member_ids = list(getattr(group, "member_track_ids", []))
        if not member_ids:
            continue
        representative_images = track_image_paths.get(member_ids[0], [])
        if not representative_images:
            continue
        label_result = label_client.score_labels(
            group_id=getattr(group, "group_id"),
            image_paths=representative_images,
            labels=["person", "vehicle", "animal", "object", "background"],
        )
        label_lines.append(
            f"- {label_result.group_id}: label={label_result.label} scores="
            + ", ".join(
                f"{score.label}:{score.score:.2f}" for score in label_result.scores[:5]
            )
        )
    if label_lines:
        existing = context.get("tool_grouping_hints", "")
        context["tool_grouping_hints"] = "\n".join(
            [
                str(existing).rstrip(),
                "",
                "Label hints:",
                *label_lines,
            ]
        ).strip()
