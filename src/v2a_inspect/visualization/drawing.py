from __future__ import annotations

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
