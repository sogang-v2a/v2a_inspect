from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from .schemas import (
    AnnotatedFrameOutput,
    DeleteSoundEventArgs,
    DeleteSoundSourceArgs,
    FrameIndexArgs,
    FrameResolutionMode,
    NoArgs,
    SceneIndexArgs,
    UpsertSoundEventArgs,
    UpsertSoundSourceArgs,
    VisualEventsArgs,
)

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor

ToolMessageContentBlock = str | dict[str, object]


def build_sound_timeline_tools(editor: SoundTimelineEditor) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            editor.list_scenes,
            args_schema=NoArgs,
            description=(
                "List all initial scenes, frame ranges, keyframes, and counts. "
                "Use once at the start to plan scene-by-scene work."
            ),
        ),
        StructuredTool.from_function(
            editor.get_scene_summary,
            args_schema=SceneIndexArgs,
            description=(
                "Primary per-scene inspection tool. Get frame range, object seeds, "
                "keyframes, and tracks before using frame images."
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
            editor.list_tracks,
            args_schema=SceneIndexArgs,
            description="List tracked visual objects inside one scene.",
        ),
        StructuredTool.from_function(
            editor.get_visual_events,
            args_schema=VisualEventsArgs,
            description=(
                "Primary visual event timeline. List computed visual events, "
                "optionally filtered by frame interval. Use before frame images."
            ),
        ),
        StructuredTool.from_function(
            editor.get_sound_timeline,
            args_schema=NoArgs,
            description="Read the current SoundTimeline. Call before final response.",
        ),
        StructuredTool.from_function(
            editor.upsert_sound_source,
            args_schema=UpsertSoundSourceArgs,
            description=(
                "Create or update a SoundSource in the SoundTimeline. Use as soon "
                "as a recurring sound emitter is known."
            ),
        ),
        StructuredTool.from_function(
            editor.delete_sound_source,
            args_schema=DeleteSoundSourceArgs,
            description="Delete a SoundSource and detach events that referenced it.",
        ),
        StructuredTool.from_function(
            editor.upsert_sound_event,
            args_schema=UpsertSoundEventArgs,
            description=(
                "Create or update a SoundEvent in the SoundTimeline. Use as soon "
                "as a sound interval is known."
            ),
        ),
        StructuredTool.from_function(
            editor.delete_sound_event,
            args_schema=DeleteSoundEventArgs,
            description="Delete a SoundEvent from the SoundTimeline.",
        ),
    ]


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
            "json": output.model_dump(mode="json", exclude={"image"}),
        },
    ]
