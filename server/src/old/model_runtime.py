from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    import torch


def _torch() -> Any:
    import torch

    return torch


def inference_device() -> "torch.device":
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the active remote inference runtime."
        )
    return torch.device("cuda")


def inference_dtype() -> "torch.dtype":
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the active remote inference runtime."
        )
    return torch.float16


def clear_cuda_cache() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_rgb_images(image_paths: Sequence[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(Path(image_path)) as image:
            images.append(image.convert("RGB"))
    return images


def move_inputs_to_device(
    inputs: dict[str, object], device: "torch.device"
) -> dict[str, object]:
    torch = _torch()
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
