from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator
from typing_extensions import Self

from .base import SchemaModel


class SoundSource(SchemaModel):
    """
    Reusable origin of sound in the editable audio plan.

    A source may correspond to a VisualObject, but it does not have to.
    """

    sound_source_id: UUID = Field(default_factory=uuid4)

    source_type: Literal[
        "visual_object",
        "scene_global",
        "offscreen_unknown",
        "non_diegetic",
    ]
    label: str

    visual_object_id: UUID | None = None

    notes: str | None = None


class SoundTrack(SchemaModel):
    """
    Reusable audible layer in the editable audio plan.

    A track is the timeline lane for one recurring sound identity, such as
    "Red Samurai sword clash" or "battlefield wind ambience".
    """

    sound_track_id: UUID = Field(default_factory=uuid4)

    track_type: Literal["dialogue", "sfx", "music", "ambience"]
    label: str
    sound_source_id: UUID | None = None
    generation_mode: Literal["tta", "vta", "hybrid", "unknown"] = "unknown"

    notes: str | None = None


class SoundEvent(SchemaModel):
    """
    One occurrence of a SoundTrack over a frame interval.
    """

    sound_event_id: UUID = Field(default_factory=uuid4)
    sound_track_id: UUID

    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)

    description: str
    notes: str | None = None


class SoundTimeline(SchemaModel):
    """
    Canonical editable audio plan for the video.

    Export formats and multitrack views should be derived from this layer.
    """

    sound_sources: list[SoundSource] = Field(default_factory=list)
    sound_tracks: list[SoundTrack] = Field(default_factory=list)
    sound_events: list[SoundEvent] = Field(default_factory=list)

    notes: str | None = None

    @model_validator(mode="after")
    def check_references(self) -> Self:
        source_ids = {source.sound_source_id for source in self.sound_sources}
        for track in self.sound_tracks:
            if (
                track.sound_source_id is not None
                and track.sound_source_id not in source_ids
            ):
                raise ValueError(f"Unknown sound_source_id: {track.sound_source_id}")

        track_ids = {track.sound_track_id for track in self.sound_tracks}
        for event in self.sound_events:
            if event.sound_track_id not in track_ids:
                raise ValueError(f"Unknown sound_track_id: {event.sound_track_id}")

        return self
