from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from v2a_inspect.client.models import Sam3TrackVideoResponse
from v2a_inspect.models import SceneTrack

from .colors import color_for_index
from .drawing import draw_bbox, draw_frame_index, draw_label
from .masks import decode_coco_rle, decode_mask_ref, overlay_mask
from .video import iter_video_frames, write_video_frames


@dataclass(frozen=True)
class _TrackPointView:
    track_id: str
    track_index: int
    bbox_xyxy: tuple[float, float, float, float] | None
    mask_rle: str | None
    mask_ref: Any | None
    confidence: float


def render_tracking_video(
    video_path: Path,
    tracks: Any,
    output_path: Path,
    *,
    start_frame_index: int | None = None,
    end_frame_index: int | None = None,
    fps: int = 30,
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> Path:
    track_list = _normalize_tracks(tracks)
    if not track_list:
        raise ValueError("tracks must contain at least one track")

    if start_frame_index is None:
        start_frame_index = min(
            point.frame_index for track in track_list for point in track.points
        )
    if end_frame_index is None:
        end_frame_index = (
            max(point.frame_index for track in track_list for point in track.points) + 1
        )

    points_by_frame = _points_by_frame(track_list)
    rendered_frames: list[Image.Image] = []
    for frame_index, frame in iter_video_frames(
        video_path,
        start_frame_index=start_frame_index,
        end_frame_index=end_frame_index,
        fps=fps,
    ):
        image = frame
        frame_points = points_by_frame.get(frame_index, [])
        for point in frame_points:
            color = color_for_index(point.track_index)
            if draw_masks:
                mask = None
                if point.mask_rle is not None:
                    mask = decode_coco_rle(point.mask_rle)
                elif point.mask_ref is not None:
                    mask = decode_mask_ref(point.mask_ref)
                if mask is not None:
                    image = overlay_mask(image, mask, color)

            if draw_boxes and point.bbox_xyxy is not None:
                draw_bbox(image, point.bbox_xyxy, color)

            if draw_labels:
                x = 10
                y = 28 + (point.track_index * 18)
                if point.bbox_xyxy is not None:
                    x = int(point.bbox_xyxy[0])
                    y = max(0, int(point.bbox_xyxy[1]) - 16)
                draw_label(
                    image, (x, y), f"{point.track_id} {point.confidence:.2f}", color
                )

        draw_frame_index(image, frame_index)
        rendered_frames.append(image)

    return write_video_frames(rendered_frames, output_path, fps=fps)


def _normalize_tracks(tracks: Any) -> list[Any]:
    if isinstance(tracks, Sam3TrackVideoResponse):
        return list(tracks.tracks)
    return list(tracks)


def _points_by_frame(tracks: Sequence[Any]) -> dict[int, list[_TrackPointView]]:
    points_by_frame: dict[int, list[_TrackPointView]] = defaultdict(list)
    for track_index, track in enumerate(tracks):
        track_id = _track_id(track)
        for point in track.points:
            points_by_frame[point.frame_index].append(
                _TrackPointView(
                    track_id=track_id,
                    track_index=track_index,
                    bbox_xyxy=getattr(point, "bbox_xyxy", None),
                    mask_rle=getattr(point, "mask_rle", None),
                    mask_ref=getattr(point, "mask", None),
                    confidence=float(point.confidence),
                )
            )
    return points_by_frame


def _track_id(track: Any) -> str:
    if isinstance(track, SceneTrack):
        return str(track.scene_track_id)[:8]
    return str(track.track_id)
