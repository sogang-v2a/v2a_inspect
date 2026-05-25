from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
from PIL import Image

from v2a_inspect.models import (
    SceneTrack,
    SceneTrackPoint,
    VideoAsset,
    VisualEvent,
    VisualObject,
)

from .colors import color_for_index
from .drawing import draw_arrow, draw_frame_index, draw_label, draw_polyline
from .masks import decode_mask_ref, overlay_mask
from .video import iter_video_frames, write_video_frames

DEFAULT_MOTION_EVENT_TYPES = {"motion", "fast_motion"}
SPARSE_MASK_COUNT = 3


@dataclass(frozen=True)
class _MotionPoint:
    frame_index: int
    centroid: tuple[float, float]
    mask: np.ndarray
    confidence: float


@dataclass(frozen=True)
class _MotionTrackView:
    event: VisualEvent
    visual_object: VisualObject
    scene_track: SceneTrack
    points: list[_MotionPoint]
    color_index: int


def render_motion_tracks_video(
    video_path: Path,
    video_asset: VideoAsset,
    output_path: Path,
    *,
    event_types: set[str] | None = None,
    start_frame_index: int | None = None,
    end_frame_index: int | None = None,
    fps: int = 30,
    draw_masks: bool = True,
    draw_centroid_path: bool = True,
    draw_motion_arrow: bool = True,
    draw_labels: bool = True,
) -> Path:
    views = _build_motion_track_views(video_asset, _event_types_or_default(event_types))
    views = _filter_views_by_frame_range(views, start_frame_index, end_frame_index)
    if not views:
        raise ValueError("video_asset does not contain matching motion events")

    render_start_frame_index, render_end_frame_index = _render_frame_range(
        views,
        start_frame_index,
        end_frame_index,
    )
    rendered_frames: list[Image.Image] = []
    for frame_index, frame in iter_video_frames(
        video_path,
        start_frame_index=render_start_frame_index,
        end_frame_index=render_end_frame_index,
        fps=fps,
    ):
        image = frame
        for view in _active_views(views, frame_index):
            image = _draw_motion_view(
                image,
                view,
                frame_index=frame_index,
                draw_masks=draw_masks,
                draw_centroid_path=draw_centroid_path,
                draw_motion_arrow=draw_motion_arrow,
                draw_labels=draw_labels,
            )
        draw_frame_index(image, frame_index)
        rendered_frames.append(image)

    return write_video_frames(rendered_frames, output_path, fps=fps)


def render_motion_tracks_image(
    video_path: Path,
    video_asset: VideoAsset,
    output_path: Path,
    *,
    event_types: set[str] | None = None,
    start_frame_index: int | None = None,
    end_frame_index: int | None = None,
    fps: int = 30,
    background_frame_index: int | None = None,
    draw_masks: bool = True,
    draw_centroid_path: bool = True,
    draw_motion_arrow: bool = True,
    draw_labels: bool = True,
) -> Path:
    views = _build_motion_track_views(video_asset, _event_types_or_default(event_types))
    views = _filter_views_by_frame_range(views, start_frame_index, end_frame_index)
    if not views:
        raise ValueError("video_asset does not contain matching motion events")

    render_start_frame_index, render_end_frame_index = _render_frame_range(
        views,
        start_frame_index,
        end_frame_index,
    )
    if background_frame_index is None:
        background_frame_index = (
            render_start_frame_index + render_end_frame_index
        ) // 2

    image = _load_frame(video_path, background_frame_index, fps=fps)
    for view in views:
        image = _draw_motion_view(
            image,
            view,
            frame_index=None,
            draw_masks=draw_masks,
            draw_centroid_path=draw_centroid_path,
            draw_motion_arrow=draw_motion_arrow,
            draw_labels=draw_labels,
        )
    draw_frame_index(image, background_frame_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _event_types_or_default(event_types: set[str] | None) -> set[str]:
    if event_types is None:
        return set(DEFAULT_MOTION_EVENT_TYPES)
    return event_types


def _build_motion_track_views(
    video_asset: VideoAsset,
    event_types: set[str],
) -> list[_MotionTrackView]:
    if video_asset.visual_identity_layer is None:
        raise ValueError("video_asset must have a visual_identity_layer")

    scene_tracks = _scene_tracks_by_id(video_asset)
    visual_objects = {
        visual_object.visual_object_id: visual_object
        for visual_object in video_asset.visual_identity_layer.visual_objects
    }

    views: list[_MotionTrackView] = []
    sorted_events = sorted(
        video_asset.visual_identity_layer.visual_events,
        key=lambda event: (event.start_frame_index, event.end_frame_index),
    )
    for color_index, event in enumerate(sorted_events):
        if event.event_type not in event_types:
            continue
        visual_object = visual_objects.get(event.visual_object_id)
        if visual_object is None:
            continue
        scene_track = _scene_track_for_visual_object(visual_object, scene_tracks)
        if scene_track is None:
            continue
        points = _motion_points_for_event(scene_track, event)
        if len(points) < 2:
            continue
        views.append(
            _MotionTrackView(
                event=event,
                visual_object=visual_object,
                scene_track=scene_track,
                points=points,
                color_index=color_index,
            )
        )
    return views


def _scene_tracks_by_id(video_asset: VideoAsset) -> dict[UUID, SceneTrack]:
    scene_tracks: dict[UUID, SceneTrack] = {}
    for initial_scene in video_asset.initial_scenes:
        for scene_track in initial_scene.scene_tracks:
            scene_tracks[scene_track.scene_track_id] = scene_track
    return scene_tracks


def _scene_track_for_visual_object(
    visual_object: VisualObject,
    scene_tracks: dict[UUID, SceneTrack],
) -> SceneTrack | None:
    for presence in visual_object.presences:
        if presence.scene_track_id is None:
            continue
        scene_track = scene_tracks.get(presence.scene_track_id)
        if scene_track is not None:
            return scene_track
    return None


def _motion_points_for_event(
    scene_track: SceneTrack,
    event: VisualEvent,
) -> list[_MotionPoint]:
    points: list[_MotionPoint] = []
    scene_track_points = sorted(scene_track.points, key=lambda point: point.frame_index)
    for point in scene_track_points:
        if point.frame_index < event.start_frame_index:
            continue
        if point.frame_index >= event.end_frame_index:
            break
        motion_point = _motion_point_from_track_point(point)
        if motion_point is not None:
            points.append(motion_point)
    return points


def _motion_point_from_track_point(point: SceneTrackPoint) -> _MotionPoint | None:
    if point.mask is None:
        return None
    mask = decode_mask_ref(point.mask)
    if not mask.any():
        return None
    centroid = _mask_centroid(mask)
    return _MotionPoint(
        frame_index=point.frame_index,
        centroid=centroid,
        mask=mask,
        confidence=point.confidence,
    )


def _filter_views_by_frame_range(
    views: list[_MotionTrackView],
    start_frame_index: int | None,
    end_frame_index: int | None,
) -> list[_MotionTrackView]:
    filtered_views: list[_MotionTrackView] = []
    for view in views:
        if (
            start_frame_index is not None
            and view.event.end_frame_index <= start_frame_index
        ):
            continue
        if (
            end_frame_index is not None
            and view.event.start_frame_index >= end_frame_index
        ):
            continue
        filtered_views.append(view)
    return filtered_views


def _render_frame_range(
    views: list[_MotionTrackView],
    start_frame_index: int | None,
    end_frame_index: int | None,
) -> tuple[int, int]:
    if start_frame_index is None:
        start_frame_index = min(view.event.start_frame_index for view in views)
    if end_frame_index is None:
        end_frame_index = max(view.event.end_frame_index for view in views)
    if end_frame_index <= start_frame_index:
        raise ValueError("end_frame_index must be greater than start_frame_index")
    return start_frame_index, end_frame_index


def _active_views(
    views: list[_MotionTrackView],
    frame_index: int,
) -> list[_MotionTrackView]:
    return [
        view
        for view in views
        if view.event.start_frame_index <= frame_index < view.event.end_frame_index
    ]


def _draw_motion_view(
    image: Image.Image,
    view: _MotionTrackView,
    *,
    frame_index: int | None,
    draw_masks: bool,
    draw_centroid_path: bool,
    draw_motion_arrow: bool,
    draw_labels: bool,
) -> Image.Image:
    color = color_for_index(view.color_index)
    points = _visible_points(view, frame_index)
    if len(points) < 2:
        return image

    if draw_masks:
        for point in _mask_points_to_draw(points, frame_index):
            image = overlay_mask(image, point.mask, color, alpha=0.25)

    centroids = [point.centroid for point in points]
    if draw_centroid_path:
        draw_polyline(image, centroids, color, width=3)
    if draw_motion_arrow:
        draw_arrow(image, centroids[0], centroids[-1], color, width=4)
    if draw_labels:
        draw_label(image, _label_position(centroids), _label_text(view), color)
    return image


def _visible_points(
    view: _MotionTrackView,
    frame_index: int | None,
) -> list[_MotionPoint]:
    if frame_index is None:
        return view.points
    return [point for point in view.points if point.frame_index <= frame_index]


def _mask_points_to_draw(
    points: list[_MotionPoint],
    frame_index: int | None,
) -> list[_MotionPoint]:
    if frame_index is not None:
        return points[-1:]
    if len(points) <= SPARSE_MASK_COUNT:
        return points
    middle_index = len(points) // 2
    return [points[0], points[middle_index], points[-1]]


def _label_position(centroids: list[tuple[float, float]]) -> tuple[int, int]:
    first_x, first_y = centroids[0]
    return max(0, int(first_x) + 6), max(0, int(first_y) - 18)


def _label_text(view: _MotionTrackView) -> str:
    label = view.visual_object.label or "object"
    return (
        f"{label} {view.event.event_type} {view.event.confidence:.2f} "
        f"{view.event.start_frame_index}-{view.event.end_frame_index}"
    )


def _load_frame(video_path: Path, frame_index: int, *, fps: int) -> Image.Image:
    for _, frame in iter_video_frames(
        video_path,
        start_frame_index=frame_index,
        end_frame_index=frame_index + 1,
        fps=fps,
    ):
        return frame
    raise ValueError(f"Could not load frame {frame_index} from {video_path}")


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    return float(cols.mean()), float(rows.mean())
