from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from v2a_inspect.contracts import TrackCrop
from v2a_inspect.tools.types import (
    FrameBatch,
    Sam3EntityTrack,
    Sam3TrackPoint,
    SampledFrame,
)


def crop_tracks(
    frame_batches: list[FrameBatch],
    tracks: list[Sam3EntityTrack],
    *,
    output_dir: str,
    padding_ratio: float = 0.1,
) -> list[TrackCrop]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    frames_by_scene = {batch.scene_index: batch.frames for batch in frame_batches}
    crops: list[TrackCrop] = []

    for track in tracks:
        scene_frames = frames_by_scene.get(track.scene_index, [])
        if not scene_frames:
            continue
        for point_index, point in enumerate(track.points):
            frame = _match_frame(scene_frames, point)
            if frame is None:
                continue
            with Image.open(frame.image_path) as image:
                crop_box = _resolve_crop_box(
                    point, image.size, padding_ratio=padding_ratio
                )
                if crop_box is None:
                    continue
                crop = image.crop(crop_box)
                crop_id = f"{track.track_id}-crop-{point_index:02d}"
                crop_dir = output_root / track.track_id
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path = crop_dir / f"{crop_id}.jpg"
                crop.save(crop_path, format="JPEG")
                crops.append(
                    TrackCrop(
                        crop_id=crop_id,
                        track_id=track.track_id,
                        scene_index=track.scene_index,
                        frame_path=frame.image_path,
                        crop_path=str(crop_path),
                        timestamp_seconds=point.timestamp_seconds,
                        bbox_xyxy=[float(value) for value in crop_box],
                        mask_rle=point.mask_rle,
                    )
                )
    return crops


def group_crop_paths_by_track(track_crops: list[TrackCrop]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for crop in track_crops:
        grouped.setdefault(crop.track_id, []).append(crop.crop_path)
    return grouped


def _match_frame(
    scene_frames: Sequence[SampledFrame],
    point: Sam3TrackPoint,
) -> SampledFrame | None:
    if not scene_frames:
        return None
    return min(
        scene_frames,
        key=lambda frame: abs(
            float(getattr(frame, "timestamp_seconds", 0.0)) - point.timestamp_seconds
        ),
    )


def _resolve_crop_box(
    point: Sam3TrackPoint,
    image_size: tuple[int, int],
    *,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    if point.mask_rle:
        mask_box = _mask_rle_to_bbox(point.mask_rle)
        if mask_box is not None:
            return _pad_box(
                mask_box, width=width, height=height, padding_ratio=padding_ratio
            )
    if point.bbox_xyxy is not None:
        left, top, right, bottom = (int(round(value)) for value in point.bbox_xyxy)
        raw_box = (left, top, right, bottom)
        return _pad_box(
            raw_box, width=width, height=height, padding_ratio=padding_ratio
        )
    return None


def _mask_rle_to_bbox(mask_rle: str) -> tuple[int, int, int, int] | None:
    try:
        payload = json.loads(mask_rle)
    except json.JSONDecodeError:
        return None
    counts = payload.get("counts")
    size = payload.get("size")
    if not isinstance(counts, list) or not isinstance(size, list) or len(size) != 2:
        return None
    height, width = int(size[0]), int(size[1])
    flat: list[int] = []
    value = 0
    for run_length in counts:
        flat.extend([value] * int(run_length))
        value = 1 - value
    if len(flat) < width * height:
        flat.extend([0] * (width * height - len(flat)))
    xs: list[int] = []
    ys: list[int] = []
    for index, pixel in enumerate(flat[: width * height]):
        if pixel:
            ys.append(index // width)
            xs.append(index % width)
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _pad_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    box_width = max(right - left, 1)
    box_height = max(bottom - top, 1)
    pad_x = max(int(round(box_width * padding_ratio)), 1)
    pad_y = max(int(round(box_height * padding_ratio)), 1)
    return (
        max(left - pad_x, 0),
        max(top - pad_y, 0),
        min(right + pad_x, width),
        min(bottom + pad_y, height),
    )
