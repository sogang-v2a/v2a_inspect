from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image


def inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def inference_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_rgb_images(image_paths: Sequence[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(Path(image_path)) as image:
            images.append(image.convert("RGB"))
    return images


def move_inputs_to_device(inputs: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
