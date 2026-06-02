from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from v2a_inspect.client.models import Sam3SegmentImageResponse

from .colors import color_for_index
from .drawing import draw_bbox, draw_label
from .masks import decode_coco_rle, overlay_mask


def render_segmented_image(
    image: Path | Image.Image,
    masks: Any,
    *,
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> Image.Image:
    rendered = _load_image(image)
    mask_list = _normalize_masks(masks)
    for mask_index, mask in enumerate(mask_list):
        color = color_for_index(mask_index)
        mask_rle = getattr(mask, "mask_rle", None)
        if draw_masks and mask_rle is not None:
            rendered = overlay_mask(rendered, decode_coco_rle(mask_rle), color)
        if draw_boxes:
            draw_bbox(rendered, mask.bbox_xyxy, color)
        if draw_labels:
            x = int(mask.bbox_xyxy[0])
            y = max(0, int(mask.bbox_xyxy[1]) - 16)
            draw_label(rendered, (x, y), f"{mask.mask_id} {mask.confidence:.2f}", color)
    return rendered


def _load_image(image: Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _normalize_masks(masks: Any) -> list[Any]:
    if isinstance(masks, Sam3SegmentImageResponse):
        return list(masks.masks)
    return list(masks)
