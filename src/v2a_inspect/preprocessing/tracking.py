from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from v2a_inspect.client import SAM3Client
from v2a_inspect.client.endpoints.base import ClientError
from v2a_inspect.client.models import (
    Sam3TextPrompt,
    Sam3Track,
    Sam3TrackPoint,
)
from v2a_inspect.media_utils import resize_coco_rle
from v2a_inspect.models import (
    InitialScene,
    MaskRef,
    ObjectSeed,
    SceneTrack,
    SceneTrackPoint,
    VideoAsset,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PromptEntry:
    object_seed: ObjectSeed
    tracking_prompt: str
    prompt: Sam3TextPrompt


async def track_initial_scene_object_seeds(
    initial_scene: InitialScene,
    *,
    video_id: str,
    sam_client: SAM3Client,
    tracking_width: int = 1280,
    tracking_height: int = 720,
    output_width: int = 1280,
    output_height: int = 720,
    seed_frame_index: int | None = None,
    score_threshold: float = 0.35,
    min_points: int = 2,
    high_confidence_threshold: float = 0.45,
    match_threshold: float = 0.45,
    min_track_mean_confidence: float = 0.0,
) -> InitialScene:
    """Track all LLM object seeds for one scene using SAM3 text prompts."""

    del seed_frame_index, score_threshold, high_confidence_threshold, match_threshold

    if initial_scene.initial_analysis is None:
        return initial_scene

    prompt_entries = _prompt_entries(initial_scene)
    return await _track_initial_scene_prompt_entries(
        initial_scene,
        video_id=video_id,
        sam_client=sam_client,
        prompt_entries=prompt_entries,
        tracking_width=tracking_width,
        tracking_height=tracking_height,
        output_width=output_width,
        output_height=output_height,
        min_points=min_points,
        min_track_mean_confidence=min_track_mean_confidence,
    )


async def _track_initial_scene_prompt_entries(
    initial_scene: InitialScene,
    *,
    video_id: str,
    sam_client: SAM3Client,
    prompt_entries: list[_PromptEntry],
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
    min_points: int,
    min_track_mean_confidence: float,
) -> InitialScene:
    scene_tracks: list[SceneTrack] = []
    if not prompt_entries:
        return initial_scene.model_copy(update={"scene_tracks": scene_tracks})

    try:
        response = await sam_client.track_video(
            video_id,
            prompts=[entry.prompt for entry in prompt_entries],
            start_frame_index=initial_scene.start_frame_index,
            end_frame_index=initial_scene.end_frame_index,
        )
    except ClientError as exc:
        if not _is_seed_tracking_failure(exc):
            raise
        logger.warning(
            "Skipping scene %s tracking after SAM3 failure: %s",
            initial_scene.initial_scene_id,
            exc,
        )
        return initial_scene.model_copy(update={"scene_tracks": scene_tracks})

    _append_scene_tracks_from_response(
        scene_tracks,
        response.tracks,
        prompt_entries=prompt_entries,
        tracking_width=tracking_width,
        tracking_height=tracking_height,
        output_width=output_width,
        output_height=output_height,
        min_points=min_points,
        min_track_mean_confidence=min_track_mean_confidence,
    )
    return initial_scene.model_copy(update={"scene_tracks": scene_tracks})


def _append_scene_tracks_from_response(
    scene_tracks: list[SceneTrack],
    sam_tracks: list[Sam3Track],
    *,
    prompt_entries: list[_PromptEntry],
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
    min_points: int,
    min_track_mean_confidence: float,
) -> None:
    entry_by_prompt_index = {
        entry.prompt.prompt_index: entry for entry in prompt_entries
    }
    for sam_track in sam_tracks:
        seed_entry = entry_by_prompt_index.get(sam_track.prompt_index)
        if seed_entry is None:
            logger.warning(
                "Skipping SAM3 track with unknown prompt_index %s",
                sam_track.prompt_index,
            )
            continue
        scene_track = _scene_track_from_sam_track(
            sam_track,
            object_seed=seed_entry.object_seed,
            tracking_prompt=seed_entry.tracking_prompt,
            tracking_width=tracking_width,
            tracking_height=tracking_height,
            output_width=output_width,
            output_height=output_height,
        )
        if scene_track is None or len(scene_track.points) < min_points:
            continue
        if scene_track.confidence < min_track_mean_confidence:
            continue
        scene_tracks.append(scene_track)


async def track_initial_scenes_object_seeds(
    video_asset: VideoAsset,
    *,
    video_id: str,
    sam_client: SAM3Client,
    scene_indexes: Iterable[int],
    score_threshold: float = 0.35,
    min_points: int = 2,
    high_confidence_threshold: float = 0.45,
    match_threshold: float = 0.45,
    min_track_mean_confidence: float = 0.0,
    segmentation_batch_size: int = 32,
    on_scene_tracked: Callable[[VideoAsset, int, int], Awaitable[VideoAsset]]
    | None = None,
) -> VideoAsset:
    """Track object seeds for selected scenes and return a new asset."""

    del (
        score_threshold,
        high_confidence_threshold,
        match_threshold,
        segmentation_batch_size,
    )

    selected_scene_indexes = set(scene_indexes)
    _validate_scene_indexes(selected_scene_indexes, len(video_asset.initial_scenes))
    tracking_width = video_asset.width
    tracking_height = video_asset.height
    if video_asset.sam3_tracking_path is not None:
        tracking_width = video_asset.sam3_tracking_width
        tracking_height = video_asset.sam3_tracking_height

    updated_scenes = list(video_asset.initial_scenes)
    tracked_scene_count = 0
    selected_scene_count = len(selected_scene_indexes)
    updated_asset = video_asset
    for scene_index, initial_scene in enumerate(video_asset.initial_scenes):
        if scene_index not in selected_scene_indexes:
            continue

        updated_scene = await _track_initial_scene_prompt_entries(
            initial_scene,
            video_id=video_id,
            sam_client=sam_client,
            prompt_entries=_prompt_entries(initial_scene),
            tracking_width=tracking_width,
            tracking_height=tracking_height,
            output_width=video_asset.width,
            output_height=video_asset.height,
            min_points=min_points,
            min_track_mean_confidence=min_track_mean_confidence,
        )
        updated_scenes[scene_index] = updated_scene
        tracked_scene_count += 1
        updated_asset = updated_asset.model_copy(
            update={"initial_scenes": updated_scenes}
        )
        if on_scene_tracked is not None:
            updated_asset = await on_scene_tracked(
                updated_asset,
                tracked_scene_count,
                selected_scene_count,
            )
            updated_scenes = list(updated_asset.initial_scenes)

    return updated_asset


def sam3_tracking_video_path(video_asset: VideoAsset) -> Path:
    """Return the video path callers should upload for SAM3 tracking."""

    if video_asset.sam3_tracking_path is not None:
        return video_asset.sam3_tracking_path
    return video_asset.source_path


def _prompt_entries(initial_scene: InitialScene) -> list[_PromptEntry]:
    if initial_scene.initial_analysis is None:
        return []
    entries = []
    for prompt_index, object_seed in enumerate(
        initial_scene.initial_analysis.object_seeds
    ):
        tracking_prompt = _tracking_prompt(object_seed)
        entries.append(
            _PromptEntry(
                object_seed=object_seed,
                tracking_prompt=tracking_prompt,
                prompt=Sam3TextPrompt(
                    prompt_index=prompt_index,
                    prompt=tracking_prompt,
                ),
            )
        )
    return entries


def _validate_scene_indexes(scene_indexes: set[int], scene_count: int) -> None:
    for scene_index in scene_indexes:
        if scene_index < 0 or scene_index >= scene_count:
            raise ValueError(f"scene index {scene_index} is outside available scenes")


def _tracking_prompt(object_seed: ObjectSeed) -> str:
    if object_seed.tracking_prompt:
        return object_seed.tracking_prompt
    return object_seed.label


def _is_seed_tracking_failure(exc: ClientError) -> bool:
    message = str(exc)
    return "HTTP 400" in message or "HTTP 422" in message


def _scene_track_from_sam_track(
    sam_track: Sam3Track,
    *,
    object_seed: ObjectSeed,
    tracking_prompt: str,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> SceneTrack | None:
    points = _scene_track_points_from_sam_points(
        sam_track.points,
        tracking_width=tracking_width,
        tracking_height=tracking_height,
        output_width=output_width,
        output_height=output_height,
    )
    if not points:
        return None

    confidence = _mean_confidence(points)
    return SceneTrack(
        source_object_seed=object_seed,
        tracking_prompt=tracking_prompt,
        points=points,
        confidence=confidence,
    )


def _scene_track_points_from_sam_points(
    sam_points: list[Sam3TrackPoint],
    *,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> list[SceneTrackPoint]:
    points: list[SceneTrackPoint] = []
    for sam_point in sam_points:
        if sam_point.confidence < 0 or sam_point.confidence > 1:
            continue

        mask = None
        if sam_point.mask_rle is not None:
            mask_rle = _scale_mask_rle(
                sam_point.mask_rle,
                tracking_width=tracking_width,
                tracking_height=tracking_height,
                output_width=output_width,
                output_height=output_height,
            )
            mask = MaskRef(rle=mask_rle)

        points.append(
            SceneTrackPoint(
                frame_index=sam_point.frame_index,
                bbox_xyxy=_scale_bbox_xyxy(
                    sam_point.bbox_xyxy,
                    tracking_width=tracking_width,
                    tracking_height=tracking_height,
                    output_width=output_width,
                    output_height=output_height,
                ),
                mask=mask,
                confidence=sam_point.confidence,
            )
        )
    return points


def _scale_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float] | None,
    *,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> tuple[float, float, float, float] | None:
    if bbox_xyxy is None:
        return None
    if tracking_width == output_width and tracking_height == output_height:
        return bbox_xyxy

    scale_x = output_width / tracking_width
    scale_y = output_height / tracking_height
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def _scale_mask_rle(
    mask_rle: str,
    *,
    tracking_width: int,
    tracking_height: int,
    output_width: int,
    output_height: int,
) -> str:
    if tracking_width == output_width and tracking_height == output_height:
        return mask_rle
    return resize_coco_rle(mask_rle, width=output_width, height=output_height)


def _mean_confidence(points: list[SceneTrackPoint]) -> float:
    total = 0.0
    for point in points:
        total += point.confidence
    return total / len(points)
