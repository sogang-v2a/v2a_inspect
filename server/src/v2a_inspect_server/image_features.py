from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageStats:
    width: int
    height: int
    brightness: float
    contrast: float
    colorfulness: float
    edge_density: float
    vertical_energy: float
    horizontal_energy: float
    histogram: tuple[float, ...]


def summarize_image_paths(image_paths: Sequence[str]) -> ImageStats:
    if not image_paths:
        return ImageStats(
            width=1,
            height=1,
            brightness=0.0,
            contrast=0.0,
            colorfulness=0.0,
            edge_density=0.0,
            vertical_energy=0.0,
            horizontal_energy=0.0,
            histogram=tuple(0.0 for _ in range(24)),
        )

    stats = [_summarize_image(Path(path)) for path in image_paths]
    return ImageStats(
        width=max(int(np.mean([item.width for item in stats])), 1),
        height=max(int(np.mean([item.height for item in stats])), 1),
        brightness=float(np.mean([item.brightness for item in stats])),
        contrast=float(np.mean([item.contrast for item in stats])),
        colorfulness=float(np.mean([item.colorfulness for item in stats])),
        edge_density=float(np.mean([item.edge_density for item in stats])),
        vertical_energy=float(np.mean([item.vertical_energy for item in stats])),
        horizontal_energy=float(np.mean([item.horizontal_energy for item in stats])),
        histogram=tuple(
            float(np.mean([item.histogram[index] for item in stats]))
            for index in range(len(stats[0].histogram))
        ),
    )


def image_embedding_vector(image_paths: Sequence[str]) -> list[float]:
    stats = summarize_image_paths(image_paths)
    return [
        stats.brightness,
        stats.contrast,
        stats.colorfulness,
        stats.edge_density,
        stats.vertical_energy,
        stats.horizontal_energy,
        *stats.histogram,
    ]


def frame_motion_score(image_paths: Sequence[str]) -> float:
    if len(image_paths) < 2:
        return 0.0
    frames = [_load_array(Path(path), size=(96, 96)) for path in image_paths[:4]]
    deltas = [
        np.mean(np.abs(frames[index + 1] - frames[index])) / 255.0
        for index in range(len(frames) - 1)
    ]
    return _clamp01(float(np.mean(deltas)))


def _summarize_image(path: Path) -> ImageStats:
    array = _load_array(path)
    height, width = array.shape[:2]
    luminance = _luminance(array)
    grad_y, grad_x = np.gradient(luminance)
    gradient_magnitude = np.sqrt((grad_x**2) + (grad_y**2))
    histogram_bins = np.linspace(0.0, 256.0, 9)
    histogram_parts: list[float] = []
    for channel_index in range(3):
        hist, _ = np.histogram(
            array[..., channel_index], bins=histogram_bins, density=True
        )
        histogram_parts.extend(float(value) for value in hist)

    return ImageStats(
        width=width,
        height=height,
        brightness=float(np.mean(luminance) / 255.0),
        contrast=float(np.std(luminance) / 128.0),
        colorfulness=float(np.std(array[..., 0] - array[..., 1]) / 128.0),
        edge_density=_clamp01(float(np.mean(gradient_magnitude > 12.0))),
        vertical_energy=_clamp01(float(np.mean(np.abs(grad_y)) / 32.0)),
        horizontal_energy=_clamp01(float(np.mean(np.abs(grad_x)) / 32.0)),
        histogram=tuple(histogram_parts),
    )


def _load_array(path: Path, *, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(size)
        return np.asarray(rgb, dtype=np.float32)


def _luminance(array: np.ndarray) -> np.ndarray:
    return (
        (0.2126 * array[..., 0]) + (0.7152 * array[..., 1]) + (0.0722 * array[..., 2])
    )


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
