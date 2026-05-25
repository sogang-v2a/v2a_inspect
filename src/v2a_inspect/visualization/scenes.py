from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from v2a_inspect.models import InitialScene, VideoAsset


def render_scene_timeline(
    video_asset: VideoAsset,
    *,
    width: int = 1200,
    row_height: int = 34,
) -> Image.Image:
    scene_count = len(video_asset.initial_scenes)
    height = max(80, 42 + scene_count * row_height)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (10, 10),
        f"{scene_count} scenes, {video_asset.frame_count} frames",
        fill="black",
    )

    timeline_left = 160
    timeline_right = width - 20
    usable_width = timeline_right - timeline_left
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        y = 38 + scene_index * row_height
        x1 = timeline_left + int(
            (scene.start_frame_index / video_asset.frame_count) * usable_width
        )
        x2 = timeline_left + int(
            (scene.end_frame_index / video_asset.frame_count) * usable_width
        )
        color = (80, 140, 220) if scene_index % 2 == 0 else (90, 170, 120)
        draw.text((10, y + 4), f"scene {scene_index:03d}", fill="black")
        draw.rectangle((x1, y, max(x1 + 1, x2), y + 22), fill=color)
        draw.text(
            (x1 + 4, y + 4),
            f"{scene.start_frame_index}-{scene.end_frame_index}",
            fill="white",
        )
    return image


def render_keyframe_grid(
    scenes: Sequence[InitialScene],
    *,
    thumbnail_width: int = 240,
    columns: int = 4,
) -> Image.Image:
    keyframes = []
    for scene_index, scene in enumerate(scenes):
        for keyframe in scene.keyframes:
            keyframes.append((scene_index, keyframe.frame_index, keyframe.image_path))

    if not keyframes:
        return Image.new("RGB", (thumbnail_width, 80), "white")

    rows = (len(keyframes) + columns - 1) // columns
    thumbnail_height = int(thumbnail_width * 9 / 16)
    label_height = 24
    image = Image.new(
        "RGB",
        (columns * thumbnail_width, rows * (thumbnail_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(image)

    for index, (scene_index, frame_index, image_path) in enumerate(keyframes):
        row = index // columns
        column = index % columns
        x = column * thumbnail_width
        y = row * (thumbnail_height + label_height)
        thumbnail = Image.open(Path(image_path)).convert("RGB")
        thumbnail.thumbnail((thumbnail_width, thumbnail_height))
        image.paste(thumbnail, (x, y))
        draw.text(
            (x + 4, y + thumbnail_height + 4),
            f"scene {scene_index}, frame {frame_index}",
            fill="black",
        )
    return image
