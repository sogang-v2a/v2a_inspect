from __future__ import annotations

from collections.abc import Sequence
from math import atan2, cos, hypot, pi, sin

from PIL import Image, ImageDraw, ImageFont

from .colors import Color


def draw_bbox(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    color: Color,
    *,
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(int(value) for value in bbox_xyxy), outline=color, width=width)


def draw_label(
    image: Image.Image,
    position: tuple[int, int],
    label: str,
    color: Color,
) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top = position
    bbox = draw.textbbox((left, top), label, font=font)
    padding = 3
    background = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )
    draw.rectangle(background, fill=color)
    draw.text((left, top), label, fill=(255, 255, 255), font=font)


def draw_frame_index(image: Image.Image, frame_index: int) -> None:
    draw_label(image, (10, 10), f"frame {frame_index}", (20, 20, 20))


def draw_polyline(
    image: Image.Image,
    points: Sequence[tuple[float, float]],
    color: Color,
    *,
    width: int = 3,
) -> None:
    if len(points) < 2:
        return

    draw = ImageDraw.Draw(image)
    draw.line(_integer_points(points), fill=color, width=width, joint="curve")


def draw_arrow(
    image: Image.Image,
    start: tuple[float, float],
    end: tuple[float, float],
    color: Color,
    *,
    width: int = 4,
) -> None:
    distance = hypot(end[0] - start[0], end[1] - start[1])
    if distance < 2:
        return

    draw = ImageDraw.Draw(image)
    start_point = (int(round(start[0])), int(round(start[1])))
    end_point = (int(round(end[0])), int(round(end[1])))
    draw.line((start_point, end_point), fill=color, width=width)

    angle = atan2(end[1] - start[1], end[0] - start[0])
    head_length = max(8, width * 3)
    left_angle = angle + pi * 0.8
    right_angle = angle - pi * 0.8
    left_point = (
        int(round(end[0] + (cos(left_angle) * head_length))),
        int(round(end[1] + (sin(left_angle) * head_length))),
    )
    right_point = (
        int(round(end[0] + (cos(right_angle) * head_length))),
        int(round(end[1] + (sin(right_angle) * head_length))),
    )
    draw.polygon((end_point, left_point, right_point), fill=color)


def _integer_points(points: Sequence[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in points]
