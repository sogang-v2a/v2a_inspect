from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from v2a_inspect.models import VideoAsset, VisualEvent, VisualObject, VisualPresence

from .colors import Color


_EVENT_COLORS: dict[str, Color] = {
    "appearance": (50, 160, 80),
    "disappearance": (45, 45, 45),
    "stationary": (150, 150, 150),
    "motion": (35, 120, 220),
    "fast_motion": (230, 85, 40),
    "scale_change": (145, 70, 200),
    "contact": (220, 60, 170),
    "uncertain": (185, 185, 185),
}
_PRESENCE_COLOR: Color = (215, 225, 235)
_PRESENCE_OUTLINE_COLOR: Color = (150, 165, 180)
_SCENE_BOUNDARY_COLOR: Color = (225, 225, 225)
_TEXT_COLOR: Color = (25, 25, 25)


@dataclass(frozen=True)
class _TimelineRow:
    visual_object: VisualObject
    presences: list[VisualPresence]
    events: list[VisualEvent]
    row_index: int


def render_visual_timeline(
    video_asset: VideoAsset,
    *,
    width: int = 1400,
    row_height: int = 30,
    show_events: bool = True,
    show_objects: bool = True,
    event_types: set[str] | None = None,
) -> Image.Image:
    """Render object presence and visual event intervals on a frame-index timeline."""

    if video_asset.visual_identity_layer is None:
        raise ValueError("video_asset must have a visual_identity_layer")
    if not show_events and not show_objects:
        raise ValueError("At least one of show_events or show_objects must be true")

    rows = _timeline_rows(video_asset, event_types)
    if not rows:
        return Image.new("RGB", (width, 80), "white")

    label_width = 220
    right_padding = 24
    top_padding = 72
    bottom_padding = 42
    timeline_left = label_width
    timeline_right = width - right_padding
    usable_width = timeline_right - timeline_left
    height = max(120, top_padding + (len(rows) * row_height) + bottom_padding)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    total_frames = max(1, video_asset.frame_count)
    event_count = sum(len(row.events) for row in rows)
    draw.text(
        (12, 12),
        f"{len(rows)} objects, {event_count} events, {video_asset.frame_count} frames",
        fill=_TEXT_COLOR,
        font=font,
    )

    _draw_frame_ticks(
        draw, total_frames, timeline_left, usable_width, top_padding, height
    )
    _draw_scene_boundaries(
        draw, video_asset, timeline_left, usable_width, top_padding, height
    )

    for row in rows:
        row_y = top_padding + (row.row_index * row_height)
        draw.text(
            (12, row_y + 7),
            _object_label(row.visual_object),
            fill=_TEXT_COLOR,
            font=font,
        )
        draw.line(
            (
                timeline_left,
                row_y + row_height - 3,
                timeline_right,
                row_y + row_height - 3,
            ),
            fill=(240, 240, 240),
        )
        if show_objects:
            _draw_presences(draw, row, total_frames, timeline_left, usable_width, row_y)
        if show_events:
            _draw_events(
                draw, row, total_frames, timeline_left, usable_width, row_y, font
            )

    _draw_legend(draw, timeline_left, 38, font)
    return image


def _timeline_rows(
    video_asset: VideoAsset,
    event_types: set[str] | None,
) -> list[_TimelineRow]:
    visual_identity_layer = video_asset.visual_identity_layer
    if visual_identity_layer is None:
        return []

    events_by_object_id: dict[object, list[VisualEvent]] = {}
    for event in visual_identity_layer.visual_events:
        if event_types is not None and event.event_type not in event_types:
            continue
        events_by_object_id.setdefault(event.visual_object_id, []).append(event)

    rows: list[_TimelineRow] = []
    visual_objects = sorted(
        visual_identity_layer.visual_objects,
        key=lambda visual_object: (
            _first_presence_frame(visual_object),
            visual_object.label or "",
            str(visual_object.visual_object_id),
        ),
    )
    for visual_object in visual_objects:
        events = sorted(
            events_by_object_id.get(visual_object.visual_object_id, []),
            key=lambda event: (
                event.start_frame_index,
                event.end_frame_index,
                event.event_type,
            ),
        )
        presences = sorted(
            visual_object.presences,
            key=lambda presence: (presence.start_frame_index, presence.end_frame_index),
        )
        if not events and not presences:
            continue
        rows.append(
            _TimelineRow(
                visual_object=visual_object,
                presences=presences,
                events=events,
                row_index=len(rows),
            )
        )
    return rows


def _draw_frame_ticks(
    draw: ImageDraw.ImageDraw,
    total_frames: int,
    timeline_left: int,
    usable_width: int,
    top_padding: int,
    height: int,
) -> None:
    tick_count = 10
    for tick_index in range(tick_count + 1):
        frame_index = round(total_frames * tick_index / tick_count)
        x = _frame_to_x(frame_index, total_frames, timeline_left, usable_width)
        draw.line((x, top_padding - 8, x, height - 28), fill=(238, 238, 238))
        draw.text((x - 10, height - 24), str(frame_index), fill=(90, 90, 90))


def _draw_scene_boundaries(
    draw: ImageDraw.ImageDraw,
    video_asset: VideoAsset,
    timeline_left: int,
    usable_width: int,
    top_padding: int,
    height: int,
) -> None:
    total_frames = max(1, video_asset.frame_count)
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        x = _frame_to_x(
            scene.start_frame_index, total_frames, timeline_left, usable_width
        )
        draw.line((x, top_padding - 24, x, height - 28), fill=_SCENE_BOUNDARY_COLOR)
        if scene_index % 2 == 0:
            draw.text(
                (x + 3, top_padding - 36), f"s{scene_index}", fill=(120, 120, 120)
            )


def _draw_presences(
    draw: ImageDraw.ImageDraw,
    row: _TimelineRow,
    total_frames: int,
    timeline_left: int,
    usable_width: int,
    row_y: int,
) -> None:
    for presence in row.presences:
        x1 = _frame_to_x(
            presence.start_frame_index, total_frames, timeline_left, usable_width
        )
        x2 = _frame_to_x(
            presence.end_frame_index, total_frames, timeline_left, usable_width
        )
        draw.rectangle(
            (x1, row_y + 8, max(x1 + 1, x2), row_y + 22),
            fill=_PRESENCE_COLOR,
            outline=_PRESENCE_OUTLINE_COLOR,
        )


def _draw_events(
    draw: ImageDraw.ImageDraw,
    row: _TimelineRow,
    total_frames: int,
    timeline_left: int,
    usable_width: int,
    row_y: int,
    font: Any,
) -> None:
    for event_index, event in enumerate(row.events):
        x1 = _frame_to_x(
            event.start_frame_index, total_frames, timeline_left, usable_width
        )
        x2 = _frame_to_x(
            event.end_frame_index, total_frames, timeline_left, usable_width
        )
        color = _event_color(event.event_type)
        y_offset = 4 if event_index % 2 == 0 else 14
        y1 = row_y + y_offset
        y2 = y1 + 8
        if x2 <= x1 + 2:
            draw.ellipse((x1 - 3, y1, x1 + 3, y2), fill=color)
            continue
        draw.rectangle((x1, y1, max(x1 + 3, x2), y2), fill=color)
        label = _event_label(event)
        text_width = _text_width(draw, label, font)
        if x2 - x1 > text_width + 8:
            draw.text((x1 + 3, y1 - 2), label, fill="white", font=font)


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font: Any) -> None:
    legend_items = [
        "appearance",
        "disappearance",
        "stationary",
        "motion",
        "fast_motion",
        "scale_change",
        "contact",
        "uncertain",
    ]
    current_x = x
    for event_type in legend_items:
        color = _event_color(event_type)
        draw.rectangle((current_x, y, current_x + 12, y + 8), fill=color)
        draw.text((current_x + 16, y - 2), event_type, fill=(70, 70, 70), font=font)
        current_x += _text_width(draw, event_type, font) + 36


def _frame_to_x(
    frame_index: int,
    total_frames: int,
    timeline_left: int,
    usable_width: int,
) -> int:
    clamped_frame_index = max(0, min(frame_index, total_frames))
    return timeline_left + int((clamped_frame_index / total_frames) * usable_width)


def _event_color(event_type: str) -> Color:
    return _EVENT_COLORS.get(event_type, (120, 120, 120))


def _event_label(event: VisualEvent) -> str:
    if event.event_type in {"appearance", "disappearance"}:
        return event.event_type[:3]
    if event.event_type == "fast_motion":
        return "fast"
    if event.event_type == "scale_change":
        return "scale"
    return event.event_type


def _object_label(visual_object: VisualObject) -> str:
    label = visual_object.label or "object"
    return f"{label} {str(visual_object.visual_object_id)[:8]}"


def _first_presence_frame(visual_object: VisualObject) -> int:
    if not visual_object.presences:
        return 0
    return min(presence.start_frame_index for presence in visual_object.presences)


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])
