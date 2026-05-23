from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from .base import SchemaModel


class VisualPresence(SchemaModel):
    """
    One interval of visual state for a video-global visual object.

    This is not a sound event. Visible does not imply sounding, and
    occluded/offscreen does not imply silent.
    """

    visual_presence_id: UUID = Field(default_factory=uuid4)

    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)

    state: Literal["visible", "occluded", "offscreen", "uncertain"]
    scene_track_id: UUID | None = None

    notes: str | None = None


class VisualObject(SchemaModel):
    """
    Video-global visual object identity built from scene-local evidence.
    """

    visual_object_id: UUID = Field(default_factory=uuid4)

    label: str | None = None
    presences: list[VisualPresence] = Field(default_factory=list)

    notes: str | None = None


class VisualEvent(SchemaModel):
    """
    Computed visual event derived from track mask/bbox time series.

    This is visual evidence only. Sound semantics are inferred later from
    VisualObjects, VisualEvents, and scene context.
    """

    visual_event_id: UUID = Field(default_factory=uuid4)

    visual_object_id: UUID
    related_visual_object_ids: list[UUID] = Field(
        default_factory=list,
        description="Other visual objects involved in this event, such as contact targets.",
    )

    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)

    event_type: Literal[
        "appearance",
        "disappearance",
        "motion",
        "fast_motion",
        "scale_change",
        "contact",
        "stationary",
        "uncertain",
    ]

    description: str | None = None
    confidence: float = Field(ge=0, le=1)

    notes: str | None = None


class VisualIdentityLayer(SchemaModel):
    """
    Cross-scene visual identity evidence and chosen visual objects.
    """

    visual_objects: list[VisualObject] = Field(default_factory=list)
    visual_events: list[VisualEvent] = Field(default_factory=list)
