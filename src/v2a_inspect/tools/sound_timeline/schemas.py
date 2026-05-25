from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from v2a_inspect.models import SoundEvent, SoundSource, VisualEvent
from v2a_inspect.models.base import SchemaModel


SoundSourceType = Literal[
    "visual_object",
    "scene_global",
    "offscreen_unknown",
    "non_diegetic",
]
SoundTrackType = Literal["dialogue", "sfx", "music", "ambience"]
SoundGenerationMode = Literal["tta", "vta", "hybrid", "unknown"]
FrameResolutionMode = Literal["low", "high"]


class NoArgs(SchemaModel):
    pass


class ListScenesArgs(SchemaModel):
    start_scene_index: int = Field(default=0, ge=0)
    limit: int = Field(default=25, ge=1, le=100)


class SceneIndexArgs(SchemaModel):
    scene_index: int = Field(ge=0)


class FrameIndexArgs(SchemaModel):
    frame_index: int = Field(ge=0)
    resolution_mode: FrameResolutionMode = "low"


class VisualEventsArgs(SchemaModel):
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, gt=0)
    limit: int = Field(default=50, ge=1, le=200)


class SoundTimelineArgs(SchemaModel):
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, gt=0)
    limit: int = Field(default=50, ge=1, le=200)


class UpsertSoundSourceArgs(SchemaModel):
    source_type: SoundSourceType
    label: str = Field(min_length=1)
    sound_source_id: UUID | None = None
    visual_object_id: UUID | None = None
    notes: str | None = None


class DeleteSoundSourceArgs(SchemaModel):
    sound_source_id: UUID


class UpsertSoundEventArgs(SchemaModel):
    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)
    description: str = Field(min_length=1)
    track_type: SoundTrackType
    sound_event_id: UUID | None = None
    sound_source_id: UUID | None = None
    generation_mode: SoundGenerationMode = "unknown"
    notes: str | None = None


class DeleteSoundEventArgs(SchemaModel):
    sound_event_id: UUID


class SceneListItem(SchemaModel):
    scene_index: int
    initial_scene_id: UUID
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    keyframe_indexes: list[int]
    track_count: int
    object_seed_count: int


class ListScenesOutput(SchemaModel):
    frame_count: int
    fps: int
    total_scene_count: int
    start_scene_index: int
    returned_scene_count: int
    next_scene_index: int | None = None
    scenes: list[SceneListItem]

    def to_tool_string(self) -> str:
        lines = [
            (
                f"scenes total={self.total_scene_count} "
                f"range={self.start_scene_index}-"
                f"{self.start_scene_index + self.returned_scene_count}"
            )
        ]
        if self.next_scene_index is not None:
            lines.append(
                "truncated: request list_scenes("
                f"start_scene_index={self.next_scene_index}) for later scenes"
            )
        for scene in self.scenes:
            labels = ""
            lines.append(
                f"- scene {scene.scene_index}: frames "
                f"{scene.start_frame_index}-{scene.end_frame_index}; "
                f"keyframes={scene.keyframe_indexes}; "
                f"tracks={scene.track_count}; object_seeds={scene.object_seed_count}"
                f"{labels}"
            )
        return "\n".join(lines)


class ObjectSeedView(SchemaModel):
    label: str
    tracking_prompt: str
    notes: str | None = None


class TrackSummary(SchemaModel):
    scene_track_id: UUID
    tracking_prompt: str
    source_label: str | None = None
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    confidence: float
    notes: str | None = None


class ListTracksOutput(SchemaModel):
    scene_index: int
    tracks: list[TrackSummary]


class SceneSummaryOutput(SchemaModel):
    scene_index: int
    initial_scene_id: UUID
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    keyframe_indexes: list[int]
    object_seeds: list[ObjectSeedView]
    tracks: list[TrackSummary]


class AnnotatedFrameTrackView(SchemaModel):
    scene_index: int
    scene_track_id: UUID
    tracking_prompt: str
    source_label: str | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    confidence: float


class AnnotatedFrameOutput(SchemaModel):
    frame_index: int
    resolution_mode: FrameResolutionMode
    width: int
    height: int
    image: str
    tracks: list[AnnotatedFrameTrackView]


class VisualEventsOutput(SchemaModel):
    total_matching_event_count: int
    start_frame_index: int | None = None
    end_frame_index: int | None = None
    limit: int
    returned_event_count: int
    visual_events: list[VisualEvent]

    def to_tool_string(self) -> str:
        frame_range = _frame_range_text(self.start_frame_index, self.end_frame_index)
        lines = [
            (
                f"visual_events frame_range={frame_range} "
                f"matching={self.total_matching_event_count} "
                f"shown={self.returned_event_count}"
            )
        ]
        if self.returned_event_count < self.total_matching_event_count:
            lines.append(
                "truncated: narrow start_frame_index/end_frame_index "
                "for more specific evidence"
            )
        for event in self.visual_events:
            lines.append(
                f"- {event.start_frame_index}-{event.end_frame_index} "
                f"{event.event_type} object={event.visual_object_id} "
                f"confidence={event.confidence:g}"
                + (f" description={event.description}" if event.description else "")
            )
        return "\n".join(lines)


class SoundTimelineViewOutput(SchemaModel):
    sound_sources: list[SoundSource]
    sound_events: list[SoundEvent]
    total_matching_event_count: int
    start_frame_index: int | None = None
    end_frame_index: int | None = None
    limit: int
    returned_event_count: int
    notes: str | None = None

    def to_tool_string(self) -> str:
        frame_range = _frame_range_text(self.start_frame_index, self.end_frame_index)
        lines = [
            (
                f"sound_timeline frame_range={frame_range} "
                f"sources={len(self.sound_sources)} "
                f"matching_events={self.total_matching_event_count} "
                f"shown_events={self.returned_event_count}"
            )
        ]
        if self.returned_event_count < self.total_matching_event_count:
            lines.append(
                "truncated: narrow start_frame_index/end_frame_index "
                "for local edit state"
            )
        if self.notes:
            lines.append(f"notes: {self.notes}")
        if self.sound_sources:
            lines.append("sources:")
            for source in self.sound_sources:
                lines.append(
                    f"- {source.sound_source_id} {source.source_type} "
                    f"label={source.label}"
                )
        if self.sound_events:
            lines.append("events:")
            for event in self.sound_events:
                lines.append(
                    f"- {event.sound_event_id} {event.start_frame_index}-"
                    f"{event.end_frame_index} {event.track_type} "
                    f"mode={event.generation_mode} description={event.description}"
                )
        return "\n".join(lines)


class DeleteSoundSourceOutput(SchemaModel):
    deleted_sound_source_id: UUID


class DeleteSoundEventOutput(SchemaModel):
    deleted_sound_event_id: UUID


def _frame_range_text(
    start_frame_index: int | None,
    end_frame_index: int | None,
) -> str:
    start = "*" if start_frame_index is None else str(start_frame_index)
    end = "*" if end_frame_index is None else str(end_frame_index)
    return f"{start}-{end}"
