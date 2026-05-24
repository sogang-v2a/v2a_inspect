from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor


def build_sound_timeline_tools(editor: SoundTimelineEditor) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            editor.list_scenes,
            description="List all initial scenes, frame ranges, keyframes, and counts.",
        ),
        StructuredTool.from_function(
            editor.get_scene_summary,
            description="Get one scene's frame range, object seeds, keyframes, and tracks.",
        ),
        StructuredTool.from_function(
            editor.get_annotated_frame,
            description="Get a video frame as a JPEG data URL with track boxes and labels.",
        ),
        StructuredTool.from_function(
            editor.list_tracks,
            description="List tracked visual objects inside one scene.",
        ),
        StructuredTool.from_function(
            editor.get_visual_events,
            description="List computed visual events, optionally filtered by frame interval.",
        ),
        StructuredTool.from_function(
            editor.get_sound_timeline,
            description="Read the current SoundTimeline.",
        ),
        StructuredTool.from_function(
            editor.upsert_sound_source,
            description="Create or update a SoundSource in the SoundTimeline.",
        ),
        StructuredTool.from_function(
            editor.delete_sound_source,
            description="Delete a SoundSource and detach events that referenced it.",
        ),
        StructuredTool.from_function(
            editor.upsert_sound_event,
            description="Create or update a SoundEvent in the SoundTimeline.",
        ),
        StructuredTool.from_function(
            editor.delete_sound_event,
            description="Delete a SoundEvent from the SoundTimeline.",
        ),
    ]
