from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from v2a_inspect.models import VisualEvent
from v2a_inspect.models.base import SchemaModel


SoundSourceType = Literal[
    "visual_object",
    "scene_global",
    "offscreen_unknown",
    "non_diegetic",
]
SoundTrackType = Literal["dialogue", "sfx", "music", "ambience"]
SoundGenerationMode = Literal["tta", "vta", "hybrid", "unknown"]


class NoArgs(SchemaModel):
    pass


class SceneIndexArgs(SchemaModel):
    scene_index: int = Field(ge=0)


class FrameIndexArgs(SchemaModel):
    frame_index: int = Field(ge=0)


class VisualEventsArgs(SchemaModel):
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, gt=0)


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
    scenes: list[SceneListItem]


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
    image: str
    tracks: list[AnnotatedFrameTrackView]


class VisualEventsOutput(SchemaModel):
    visual_events: list[VisualEvent]


class DeleteSoundSourceOutput(SchemaModel):
    deleted_sound_source_id: UUID


class DeleteSoundEventOutput(SchemaModel):
    deleted_sound_event_id: UUID
