from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from v2a_inspect.models import MaskRef

from .colors import Color


def decode_coco_rle(mask_rle: str) -> np.ndarray:
    payload = json.loads(mask_rle)
    if payload.get("encoding") != "coco_rle":
        raise ValueError(f"Unsupported mask encoding: {payload.get('encoding')}")

    rle = {
        "size": payload["size"],
        "counts": payload["counts"].encode("ascii"),
    }
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def decode_mask_ref(mask: MaskRef) -> np.ndarray:
    if mask.rle is not None:
        return decode_coco_rle(mask.rle)
    if mask.path is None:
        raise ValueError("MaskRef must contain either rle or path")

    image = Image.open(mask.path).convert("L")
    return np.array(image) > 0


def load_mask_path(mask_path: Path) -> np.ndarray:
    image = Image.open(mask_path).convert("L")
    return np.array(image) > 0


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: Color,
    *,
    alpha: float = 0.45,
) -> Image.Image:
    base = image.convert("RGBA")
    mask_image = Image.fromarray((mask.astype(np.uint8) * int(255 * alpha)), mode="L")
    color_image = Image.new("RGBA", base.size, (*color, 0))
    color_image.putalpha(mask_image)
    return Image.alpha_composite(base, color_image).convert("RGB")
