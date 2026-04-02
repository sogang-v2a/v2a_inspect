from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from v2a_inspect.settings import settings
from v2a_inspect.tools import (
    FrameBatch,
    Sam3EntityTrack,
    Sam3VisualFeatures,
    TrackRoutingDecision,
    aggregate_group_routes,
    detect_scenes,
    group_entity_embeddings,
    probe_video,
    route_track,
    sample_frames,
)
from v2a_inspect.workflows import InspectOptions

if TYPE_CHECKING:
    from v2a_inspect_server.runtime import ToolingRuntime


class _LabelClientLike(Protocol):
    def score_image_labels(
        self,
        *,
        image_paths: list[str],
        labels: list[str],
    ) -> Sequence["_LabelScoreLike"]: ...

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
            options=options,
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
    options: InspectOptions,
) -> None:
    raw_progress = context.get("progress_messages", [])
    progress_messages = (
        [str(item) for item in raw_progress] if isinstance(raw_progress, list) else []
    )
    try:
        label_client = getattr(tooling_runtime, "label_client", None)
        scene_prompts = (
            _scene_prompt_candidates(frame_batches, label_client)
            if label_client is not None
            else {}
        )
        sam3_track_set = tooling_runtime.sam3_client.extract_entities(
            list(frame_batches),
            prompts_by_scene=scene_prompts,
        )
        context["sam3_track_set"] = sam3_track_set
        normalized_tracks = _normalize_tracks(getattr(sam3_track_set, "tracks", []))
        context["tool_grouping_hints"] = _grouping_hints_from_tracks(normalized_tracks)
        progress_messages.append(
            f"Tool pipeline: extracted {len(normalized_tracks)} SAM3 tracks."
        )

        tracks_by_id = {track.track_id: track for track in normalized_tracks}
        track_image_paths = _track_image_paths(
            frame_batches=frame_batches,
            tracks=normalized_tracks,
        )
        track_routing_decisions = _build_track_routing_decisions(normalized_tracks)
        group_objects: Sequence[object] = _singleton_groups_for_tracks(
            normalized_tracks
        )

        if track_image_paths:
            embedding_client = getattr(tooling_runtime, "embedding_client", None)
            if embedding_client is not None and label_client is not None:
                embeddings = embedding_client.embed_images(track_image_paths)
                context["entity_embeddings"] = embeddings
                candidate_groups = group_entity_embeddings(
                    embeddings,
                    tracks_by_id=tracks_by_id,
                )
                group_objects = candidate_groups.groups
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

        routing_decisions = _build_group_routing_decisions(
            group_objects=group_objects,
            track_decisions_by_track_id=track_routing_decisions,
        )
        if routing_decisions:
            routing_hints = _routing_hints_from_decisions(
                routing_decisions,
                track_decisions_by_track_id=track_routing_decisions,
            )
            context["routing_decisions"] = routing_decisions
            context["tool_routing_hints"] = routing_hints
            context["tool_grouping_hints"] = _append_hint_section(
                context.get("tool_grouping_hints", ""),
                routing_hints,
            )

        verify_hints = _verify_hints_from_groups(
            group_objects=group_objects,
            tracks_by_id=tracks_by_id,
            minimum_group_size=2 if options.enable_vlm_verify else 1,
        )
        if verify_hints:
            context["tool_verify_hints"] = verify_hints
            context["tool_grouping_hints"] = _append_hint_section(
                context.get("tool_grouping_hints", ""),
                verify_hints,
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


def _scene_prompt_candidates(
    frame_batches: Sequence[FrameBatch],
    label_client: _LabelClientLike,
) -> dict[int, list[str]]:
    candidate_labels = [
        "person",
        "man",
        "woman",
        "child",
        "cat",
        "dog",
        "vehicle",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
        "boat",
        "animal",
        "bird",
        "tree",
        "plant",
        "building",
        "screen",
        "laptop",
        "object",
        "background",
    ]
    prompts_by_scene: dict[int, list[str]] = {}
    for batch in frame_batches:
        image_paths = [frame.image_path for frame in batch.frames]
        if not image_paths:
            continue
        scores = list(
            label_client.score_image_labels(
                image_paths=image_paths,
                labels=candidate_labels,
            )
        )
        prompts = [
            score.label
            for score in scores
            if score.label != "background" and score.score >= 0.2
        ][:3]
        if not prompts:
            prompts = ["object"]
        prompts_by_scene[batch.scene_index] = prompts
    return prompts_by_scene


def _normalize_tracks(tracks: Sequence[object]) -> list[Sam3EntityTrack]:
    normalized: list[Sam3EntityTrack] = []
    for track in tracks:
        normalized_track = _normalize_track(track)
        if normalized_track is not None:
            normalized.append(normalized_track)
    return normalized


def _normalize_track(track: object) -> Sam3EntityTrack | None:
    track_id = getattr(track, "track_id", None)
    if track_id is None:
        return None
    scene_index = _safe_int(getattr(track, "scene_index", 0))
    start_seconds = _safe_float(getattr(track, "start_seconds", 0.0))
    end_seconds = _safe_float(getattr(track, "end_seconds", start_seconds))
    features = getattr(track, "features", None)
    return Sam3EntityTrack(
        track_id=str(track_id),
        scene_index=max(scene_index, 0),
        start_seconds=max(start_seconds, 0.0),
        end_seconds=max(end_seconds, start_seconds),
        confidence=_clamp01(getattr(track, "confidence", 0.0)),
        label_hint=getattr(track, "label_hint", None),
        features=Sam3VisualFeatures(
            motion_score=_feature_value(features, "motion_score"),
            interaction_score=_feature_value(features, "interaction_score"),
            crowd_score=_feature_value(features, "crowd_score"),
            camera_dynamics_score=_feature_value(features, "camera_dynamics_score"),
        ),
    )


def _feature_value(features: object, name: str) -> float:
    return _clamp01(getattr(features, name, 0.0))


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _safe_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _clamp01(value: object) -> float:
    return min(max(_safe_float(value), 0.0), 1.0)


def _group_member_ids(group: object) -> list[str]:
    raw_members: object
    if hasattr(group, "member_track_ids"):
        raw_members = getattr(group, "member_track_ids")
    elif isinstance(group, Mapping):
        mapping = cast(Mapping[str, object], group)
        raw_members = mapping.get("member_track_ids", [])
    else:
        raw_members = []
    if not isinstance(raw_members, Sequence) or isinstance(raw_members, (str, bytes)):
        return []
    return [str(member_id) for member_id in raw_members]


def _group_id(group: object) -> str:
    if hasattr(group, "group_id"):
        return str(getattr(group, "group_id"))
    if isinstance(group, Mapping):
        mapping = cast(Mapping[str, object], group)
        return str(mapping.get("group_id", "group"))
    return "group"


def _group_confidence(group: object) -> float:
    if hasattr(group, "confidence"):
        return _safe_float(getattr(group, "confidence"))
    if isinstance(group, Mapping):
        mapping = cast(Mapping[str, object], group)
        return _safe_float(mapping.get("confidence", 0.0))
    return 0.0


def _grouping_hints_from_tracks(tracks: Sequence[Sam3EntityTrack]) -> str:
    if not tracks:
        return "SAM3 found no tracks."
    lines = []
    for track in tracks[:20]:
        lines.append(
            f"- {track.track_id}: scene={track.scene_index} "
            f"range={track.start_seconds:.1f}-{track.end_seconds:.1f} "
            f"label_hint={track.label_hint} confidence={track.confidence:.2f}"
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
        member_ids = _group_member_ids(group)
        member_summaries = []
        for member_id in member_ids:
            track = tracks_by_id.get(member_id)
            if track is None:
                continue
            member_summaries.append(
                f"{member_id}(scene={track.scene_index}, conf={track.confidence:.2f}, hint={track.label_hint})"
            )
        lines.append(
            f"- {_group_id(group)}: members={', '.join(member_summaries)} confidence={_group_confidence(group):.2f}"
        )
    return "\n".join(["Embedding/SAM3 grouping hints:", *lines])


def _build_track_routing_decisions(
    tracks: Sequence[Sam3EntityTrack],
) -> dict[str, TrackRoutingDecision]:
    return {track.track_id: route_track(track) for track in tracks}


def _singleton_groups_for_tracks(tracks: Sequence[Sam3EntityTrack]) -> list[object]:
    return [
        {"group_id": f"track:{track.track_id}", "member_track_ids": [track.track_id]}
        for track in tracks
    ]


def _build_group_routing_decisions(
    *,
    group_objects: Sequence[object],
    track_decisions_by_track_id: dict[str, TrackRoutingDecision],
) -> list[object]:
    routing_decisions: list[object] = []
    for group in group_objects:
        group_id = _group_id(group)
        member_track_ids = _group_member_ids(group)
        routing_decisions.append(
            aggregate_group_routes(
                group_id=group_id,
                member_track_ids=member_track_ids,
                decisions_by_track_id=track_decisions_by_track_id,
            )
        )
    return routing_decisions


def _routing_hints_from_decisions(
    routing_decisions: Sequence[object],
    *,
    track_decisions_by_track_id: dict[str, TrackRoutingDecision],
) -> str:
    if not routing_decisions:
        return ""
    lines = ["Routing/model-selection hints:"]
    for decision in routing_decisions:
        member_track_ids = list(getattr(decision, "member_track_ids", []))
        member_summaries = []
        for track_id in member_track_ids:
            track_decision = track_decisions_by_track_id.get(track_id)
            if track_decision is None:
                member_summaries.append(track_id)
            else:
                member_summaries.append(
                    f"{track_id}:{track_decision.model_type}@{track_decision.confidence:.2f}"
                )
        lines.append(
            f"- {_group_id(decision)}: recommend={getattr(decision, 'model_type', 'TTA')} "
            f"confidence={_safe_float(getattr(decision, 'confidence', 0.0)):.2f} members={', '.join(member_summaries) or 'none'}"
        )
    return "\n".join(lines)


def _verify_hints_from_groups(
    *,
    group_objects: Sequence[object],
    tracks_by_id: dict[str, Sam3EntityTrack],
    minimum_group_size: int,
) -> str:
    if not group_objects:
        return ""
    lines = ["Verify/group hints:"]
    for group in group_objects:
        group_id = _group_id(group)
        member_track_ids = _group_member_ids(group)
        member_tracks = [
            tracks_by_id[track_id]
            for track_id in member_track_ids
            if track_id in tracks_by_id
        ]
        scene_ids = sorted({track.scene_index for track in member_tracks})
        label_hints = sorted(
            {track.label_hint for track in member_tracks if track.label_hint}
        )
        if len(member_track_ids) < minimum_group_size:
            priority = "low"
            recommendation = "singleton candidate; keep as-is unless Gemini has contradictory evidence"
        elif len(scene_ids) > 1:
            priority = "high"
            recommendation = "cross-scene cluster; ask Gemini to confirm same-entity consistency before merging"
        else:
            priority = "medium"
            recommendation = "same-scene cluster; Gemini should verify grouping with local visual differences"
        lines.append(
            f"- {group_id}: priority={priority} scenes={scene_ids or ['unknown']} labels={label_hints or ['unknown']} "
            f"members={', '.join(member_track_ids) or 'none'} -> {recommendation}"
        )
    return "\n".join(lines)


def _append_hint_section(existing: object, section: str) -> str:
    if not section:
        return str(existing).strip()
    existing_text = str(existing).strip()
    if not existing_text:
        return section
    return "\n\n".join([existing_text, section])


def _append_group_label_hints(
    context: dict[str, object],
    *,
    label_client: _LabelClientLike,
    candidate_groups: Sequence[object],
    track_image_paths: dict[str, list[str]],
) -> None:
    label_lines: list[str] = []
    for group in candidate_groups:
        member_ids = _group_member_ids(group)
        if not member_ids:
            continue
        representative_images = track_image_paths.get(member_ids[0], [])
        if not representative_images:
            continue
        label_result = label_client.score_labels(
            group_id=_group_id(group),
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
