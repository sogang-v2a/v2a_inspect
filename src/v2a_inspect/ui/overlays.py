from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from v2a_inspect.models import SceneTrack, VideoAsset
from v2a_inspect.visualization.colors import color_for_index
from v2a_inspect.visualization.drawing import draw_bbox, draw_label
from v2a_inspect.visualization.masks import decode_mask_ref

OVERLAY_SIZE = (1280, 720)


def render_tracking_overlay(video_asset: VideoAsset, frame_index: int) -> bytes:
    image = Image.new("RGBA", OVERLAY_SIZE, (0, 0, 0, 0))
    for track_index, track in enumerate(_tracks(video_asset)):
        point = next(
            (item for item in track.points if item.frame_index == frame_index),
            None,
        )
        if point is None:
            continue

        color = color_for_index(track_index)
        if point.mask is not None:
            mask = decode_mask_ref(point.mask)
            image = _overlay_mask(image, mask, color)

        if point.bbox_xyxy is not None:
            draw_bbox(image, point.bbox_xyxy, color, width=3)

        label = _track_label(track)
        if point.bbox_xyxy is None:
            position = (10, 24 + (track_index * 18))
        else:
            position = (
                int(point.bbox_xyxy[0]),
                max(0, int(point.bbox_xyxy[1]) - 16),
            )
        draw_label(image, position, f"{label} {point.confidence:.2f}", color)

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _tracks(video_asset: VideoAsset) -> list[SceneTrack]:
    return [
        track for scene in video_asset.initial_scenes for track in scene.scene_tracks
    ]


def _track_label(track: SceneTrack) -> str:
    if track.source_object_seed is not None:
        return track.source_object_seed.label
    return track.tracking_prompt


def _overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    base = image.convert("RGBA")
    mask_image = Image.fromarray((mask.astype(np.uint8) * 95), mode="L")
    color_image = Image.new("RGBA", base.size, (*color, 0))
    color_image.putalpha(mask_image)
    return Image.alpha_composite(base, color_image)
