from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

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


class SoundEvent(SchemaModel):
    """
    One sound occurrence over a frame interval.

    This is the main editable unit of the SoundTimeline.
    """

    sound_event_id: UUID = Field(default_factory=uuid4)

    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)

    description: str
    track_type: Literal["dialogue", "sfx", "music", "ambience"]
    sound_source_id: UUID | None = None
    generation_mode: Literal["tta", "vta", "hybrid", "unknown"] = "unknown"

    notes: str | None = None


class SoundTimeline(SchemaModel):
    """
    Canonical editable audio plan for the video.

    Export formats and multitrack views should be derived from this layer.
    """

    sound_sources: list[SoundSource] = Field(default_factory=list)
    sound_events: list[SoundEvent] = Field(default_factory=list)

    notes: str | None = None
