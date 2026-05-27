from __future__ import annotations

from collections.abc import Callable
import json
from typing import TYPE_CHECKING
from uuid import UUID

from langchain_core.tools import StructuredTool

from v2a_inspect.models import SchemaModel, SoundEvent, SoundSource

from .schemas import (
    AnnotatedFrameOutput,
    DeleteSoundEventArgs,
    DeleteSoundSourceArgs,
    FrameIndexArgs,
    FrameResolutionMode,
    ListScenesArgs,
    SceneIndexArgs,
    SoundGenerationMode,
    SoundSourceType,
    SoundTimelineArgs,
    SoundTrackType,
    UpsertSoundEventArgs,
    UpsertSoundSourceArgs,
    VisualEventsArgs,
    disambiguate_labels,
)

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor

ToolMessageContentBlock = str | dict[str, object]


def build_sound_timeline_tools(editor: SoundTimelineEditor) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            _make_list_scenes_tool(editor),
            args_schema=ListScenesArgs,
            description=(
                "List a page of initial scenes, frame ranges, keyframes, and "
                "counts. Use start_scene_index and limit for the next small "
                "batch only; do not page ahead before writing timeline edits."
            ),
        ),
        StructuredTool.from_function(
            _make_get_scene_summary_tool(editor),
            args_schema=SceneIndexArgs,
            description=(
                "Primary per-scene inspection tool. Get frame range, object seeds, "
                "keyframes, and tracks before using frame images. Use only for "
                "the current scene batch; do not call repeatedly without writing events."
            ),
        ),
        StructuredTool.from_function(
            _make_annotated_frame_tool(editor),
            args_schema=FrameIndexArgs,
            description=(
                "Targeted visual confirmation only. Returns LangChain message "
                "blocks: text, image, and JSON track metadata. Use sparingly "
                "after scene summary and visual events. Use resolution_mode='low' "
                "by default; use 'high' only when low-res lacks needed detail."
            ),
        ),
        StructuredTool.from_function(
            _make_list_tracks_tool(editor),
            args_schema=SceneIndexArgs,
            description="List tracked visual objects inside one scene.",
        ),
        StructuredTool.from_function(
            _make_get_visual_events_tool(editor),
            args_schema=VisualEventsArgs,
            description=(
                "Primary visual event timeline. List bounded computed visual "
                "events for the current scene or batch frame window. If truncated, "
                "narrow the frame interval."
            ),
        ),
        StructuredTool.from_function(
            _make_get_sound_timeline_tool(editor),
            args_schema=SoundTimelineArgs,
            description=(
                "Read a bounded view of the current SoundTimeline. Use frame "
                "windows after editing each batch before inspecting later scenes."
            ),
        ),
        StructuredTool.from_function(
            _make_upsert_sound_source_tool(editor),
            args_schema=UpsertSoundSourceArgs,
            description=(
                "Create or update a SoundSource in the SoundTimeline. Use as soon "
                "as a recurring sound emitter is known."
            ),
        ),
        StructuredTool.from_function(
            _make_delete_sound_source_tool(editor),
            args_schema=DeleteSoundSourceArgs,
            description="Delete a SoundSource and detach events that referenced it.",
        ),
        StructuredTool.from_function(
            _make_upsert_sound_event_tool(editor),
            args_schema=UpsertSoundEventArgs,
            description=(
                "Create or update a SoundEvent in the SoundTimeline. Use as soon "
                "as a sound interval is known and before moving to later scenes."
            ),
        ),
        StructuredTool.from_function(
            _make_delete_sound_event_tool(editor),
            args_schema=DeleteSoundEventArgs,
            description="Delete a SoundEvent from the SoundTimeline.",
        ),
    ]


def _tool_string(value: object) -> str:
    if isinstance(value, SoundSource):
        return f"sound_source {value.source_type} label={value.label}"
    if isinstance(value, SoundEvent):
        source = "" if value.sound_source_id is None else " source=linked"
        return (
            f"sound_event {value.start_frame_index}-{value.end_frame_index} "
            f"{value.track_type} mode={value.generation_mode}{source} "
            f"description={value.description}"
        )
    if isinstance(value, SchemaModel):
        return value.to_tool_string()
    return json.dumps(value, indent=2, default=str)


def _make_list_scenes_tool(editor: SoundTimelineEditor) -> Callable[[int, int], str]:
    def list_scenes(start_scene_index: int = 0, limit: int = 25) -> str:
        return _tool_string(editor.list_scenes(start_scene_index, limit))

    return list_scenes


def _make_get_scene_summary_tool(editor: SoundTimelineEditor) -> Callable[[int], str]:
    def get_scene_summary(scene_index: int) -> str:
        return _tool_string(editor.get_scene_summary(scene_index))

    return get_scene_summary


def _make_list_tracks_tool(editor: SoundTimelineEditor) -> Callable[[int], str]:
    def list_tracks(scene_index: int) -> str:
        return _tool_string(editor.list_tracks(scene_index))

    return list_tracks


def _make_get_visual_events_tool(
    editor: SoundTimelineEditor,
) -> Callable[[int | None, int | None, int], str]:
    def get_visual_events(
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 50,
    ) -> str:
        return _tool_string(
            editor.get_visual_events(
                start_frame_index,
                end_frame_index,
                limit,
            )
        )

    return get_visual_events


def _make_get_sound_timeline_tool(
    editor: SoundTimelineEditor,
) -> Callable[[int | None, int | None, int], str]:
    def get_sound_timeline(
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 50,
    ) -> str:
        return _tool_string(
            editor.get_sound_timeline(
                start_frame_index,
                end_frame_index,
                limit,
            )
        )

    return get_sound_timeline


def _make_upsert_sound_source_tool(
    editor: SoundTimelineEditor,
) -> Callable[
    [SoundSourceType, str, UUID | None, UUID | None, str | None],
    str,
]:
    def upsert_sound_source(
        source_type: SoundSourceType,
        label: str,
        sound_source_id: UUID | None = None,
        visual_object_id: UUID | None = None,
        notes: str | None = None,
    ) -> str:
        return _tool_string(
            editor.upsert_sound_source(
                source_type=source_type,
                label=label,
                sound_source_id=sound_source_id,
                visual_object_id=visual_object_id,
                notes=notes,
            )
        )

    return upsert_sound_source


def _make_delete_sound_source_tool(
    editor: SoundTimelineEditor,
) -> Callable[[UUID], str]:
    def delete_sound_source(sound_source_id: UUID) -> str:
        return _tool_string(editor.delete_sound_source(sound_source_id))

    return delete_sound_source


def _make_upsert_sound_event_tool(
    editor: SoundTimelineEditor,
) -> Callable[
    [
        int,
        int,
        str,
        SoundTrackType,
        UUID | None,
        UUID | None,
        SoundGenerationMode,
        str | None,
    ],
    str,
]:
    def upsert_sound_event(
        start_frame_index: int,
        end_frame_index: int,
        description: str,
        track_type: SoundTrackType,
        sound_event_id: UUID | None = None,
        sound_source_id: UUID | None = None,
        generation_mode: SoundGenerationMode = "unknown",
        notes: str | None = None,
    ) -> str:
        return _tool_string(
            editor.upsert_sound_event(
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                description=description,
                track_type=track_type,
                sound_event_id=sound_event_id,
                sound_source_id=sound_source_id,
                generation_mode=generation_mode,
                notes=notes,
            )
        )

    return upsert_sound_event


def _make_delete_sound_event_tool(editor: SoundTimelineEditor) -> Callable[[UUID], str]:
    def delete_sound_event(sound_event_id: UUID) -> str:
        return _tool_string(editor.delete_sound_event(sound_event_id))

    return delete_sound_event


def _make_annotated_frame_tool(
    editor: SoundTimelineEditor,
) -> Callable[[int, FrameResolutionMode], list[ToolMessageContentBlock]]:
    def get_annotated_frame_message(
        frame_index: int,
        resolution_mode: FrameResolutionMode = "low",
    ) -> list[ToolMessageContentBlock]:
        output = editor.get_annotated_frame(frame_index, resolution_mode)
        return _to_annotated_frame_message_blocks(output)

    return get_annotated_frame_message


def _to_annotated_frame_message_blocks(
    output: AnnotatedFrameOutput,
) -> list[ToolMessageContentBlock]:
    track_count = len(output.tracks)
    track_labels = disambiguate_labels([track.display_label for track in output.tracks])
    return [
        {
            "type": "text",
            "text": (
                f"Annotated frame {output.frame_index} with "
                f"{track_count} visible tracks."
            ),
        },
        {
            "type": "image",
            "base64": output.image,
            "mime_type": "image/jpeg",
        },
        {
            "type": "json",
            "json": {
                "frame_index": output.frame_index,
                "resolution_mode": output.resolution_mode,
                "width": output.width,
                "height": output.height,
                "tracks": [
                    {
                        "scene_index": track.scene_index,
                        "label": label,
                        "bbox_xyxy": _rounded_bbox(track.bbox_xyxy),
                        "confidence": round(track.confidence, 2),
                    }
                    for label, track in zip(
                        track_labels,
                        output.tracks,
                        strict=True,
                    )
                ],
            },
        },
    ]


def _rounded_bbox(
    bbox_xyxy: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int] | None:
    if bbox_xyxy is None:
        return None
    return (
        round(bbox_xyxy[0]),
        round(bbox_xyxy[1]),
        round(bbox_xyxy[2]),
        round(bbox_xyxy[3]),
    )
