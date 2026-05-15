from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field

from .base import SchemaModel


class TrackLinkCandidate(SchemaModel):
    """
    DINO-generated evidence that two SceneTracks may represent the same object.

    This is evidence, not a visual identity decision.
    """

    track_link_candidate_id: UUID = Field(default_factory=uuid4)

    left_scene_track_id: UUID
    right_scene_track_id: UUID

    similarity: float = Field(ge=0, le=1)

    notes: str | None = None


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


class VisualIdentityLayer(SchemaModel):
    """
    Cross-scene visual identity evidence and chosen visual objects.
    """

    track_link_candidates: list[TrackLinkCandidate] = Field(default_factory=list)
    visual_objects: list[VisualObject] = Field(default_factory=list)
