from __future__ import annotations

import json

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


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


def encode_coco_rle(mask: np.ndarray) -> str:
    encoded_mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
    return json.dumps(
        {
            "encoding": "coco_rle",
            "size": encoded_mask["size"],
            "counts": encoded_mask["counts"],
        },
        separators=(",", ":"),
    )


def resize_mask_nearest(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = image.resize((width, height), Image.Resampling.NEAREST)
    return np.array(resized) > 0


def resize_coco_rle(mask_rle: str, *, width: int, height: int) -> str:
    mask = decode_coco_rle(mask_rle)
    resized = resize_mask_nearest(mask, width=width, height=height)
    return encode_coco_rle(resized)
