from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from v2a_inspect.models import SoundEvent, SoundSource
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
    object_labels: list[str] = Field(default_factory=list)


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
            labels = _compact_list(scene.object_labels)
            lines.append(
                f"- scene {scene.scene_index}: frames "
                f"{scene.start_frame_index}-{scene.end_frame_index}; "
                f"keyframes={scene.keyframe_indexes}; "
                f"tracks={scene.track_count}; object_seeds={scene.object_seed_count}"
                + (f"; objects={labels}" if labels else "")
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

    @property
    def display_label(self) -> str:
        return self.source_label or self.tracking_prompt


class ListTracksOutput(SchemaModel):
    scene_index: int
    tracks: list[TrackSummary]

    def to_tool_string(self) -> str:
        lines = [f"scene {self.scene_index} tracks={len(self.tracks)}"]
        for track_number, track in enumerate(self.tracks, start=1):
            lines.append(_track_line(track_number, track))
        return "\n".join(lines)


class SceneSummaryOutput(SchemaModel):
    scene_index: int
    initial_scene_id: UUID
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    keyframe_indexes: list[int]
    object_seeds: list[ObjectSeedView]
    tracks: list[TrackSummary]

    def to_tool_string(self) -> str:
        lines = [
            (
                f"scene {self.scene_index}: frames "
                f"{self.start_frame_index}-{self.end_frame_index}; "
                f"keyframes={self.keyframe_indexes}"
            )
        ]
        if self.object_seeds:
            labels = _compact_list([seed.label for seed in self.object_seeds])
            lines.append(f"objects={labels}")
            for seed in self.object_seeds:
                detail = f"- object {seed.label}: prompt={seed.tracking_prompt}"
                if seed.notes:
                    detail += f"; notes={seed.notes}"
                lines.append(detail)
        if self.tracks:
            lines.append(f"tracks={len(self.tracks)}")
            for track_number, track in enumerate(self.tracks, start=1):
                lines.append(_track_line(track_number, track))
        return "\n".join(lines)


class AnnotatedFrameTrackView(SchemaModel):
    scene_index: int
    scene_track_id: UUID
    tracking_prompt: str
    source_label: str | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    confidence: float

    @property
    def display_label(self) -> str:
        return self.source_label or self.tracking_prompt


class AnnotatedFrameOutput(SchemaModel):
    frame_index: int
    resolution_mode: FrameResolutionMode
    width: int
    height: int
    image: str
    tracks: list[AnnotatedFrameTrackView]


class VisualEventView(SchemaModel):
    object_label: str
    related_object_labels: list[str] = Field(default_factory=list)
    start_frame_index: int
    end_frame_index: int
    event_type: str
    description: str | None = None
    confidence: float
    notes: str | None = None


class VisualEventsOutput(SchemaModel):
    total_matching_event_count: int
    start_frame_index: int | None = None
    end_frame_index: int | None = None
    limit: int
    returned_event_count: int
    visual_events: list[VisualEventView]

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
            related = _compact_list(event.related_object_labels)
            lines.append(
                f"- {event.start_frame_index}-{event.end_frame_index} "
                f"{event.event_type} object={event.object_label} "
                f"confidence={event.confidence:g}"
                + (f" related={related}" if related else "")
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
            for source_number, source in enumerate(self.sound_sources, start=1):
                lines.append(
                    f"- source {source_number}: {source.source_type} label={source.label}"
                )
        if self.sound_events:
            lines.append("events:")
            for event_number, event in enumerate(self.sound_events, start=1):
                source_label = _sound_source_label(event, self.sound_sources)
                lines.append(
                    f"- event {event_number}: {event.start_frame_index}-"
                    f"{event.end_frame_index} {event.track_type} "
                    f"mode={event.generation_mode}"
                    + (f" source={source_label}" if source_label else "")
                    + f" description={event.description}"
                )
        return "\n".join(lines)


class DeleteSoundSourceOutput(SchemaModel):
    deleted_sound_source_id: UUID

    def to_tool_string(self) -> str:
        return "deleted sound source"


class DeleteSoundEventOutput(SchemaModel):
    deleted_sound_event_id: UUID

    def to_tool_string(self) -> str:
        return "deleted sound event"


def _track_line(track_number: int, track: TrackSummary) -> str:
    line = (
        f"- track {track_number}: {track.display_label}; "
        f"frames={track.start_frame_index}-{track.end_frame_index}; "
        f"confidence={track.confidence:g}"
    )
    if track.notes:
        line += f"; notes={track.notes}"
    return line


def _compact_list(items: list[str]) -> str:
    values = [item for item in items if item]
    return "[" + ", ".join(values) + "]" if values else ""


def _sound_source_label(
    event: SoundEvent,
    sound_sources: list[SoundSource],
) -> str | None:
    if event.sound_source_id is None:
        return None
    for source in sound_sources:
        if source.sound_source_id == event.sound_source_id:
            return source.label
    return None


def _frame_range_text(
    start_frame_index: int | None,
    end_frame_index: int | None,
) -> str:
    start = "*" if start_frame_index is None else str(start_frame_index)
    end = "*" if end_frame_index is None else str(end_frame_index)
    return f"{start}-{end}"
