from pathlib import Path
from uuid import UUID, uuid4

from pydantic import Field

from .base import SchemaModel


class Keyframe(SchemaModel):
    """
    Representative frame sampled from an InitialScene.
    """

    keyframe_id: UUID = Field(default_factory=uuid4)

    frame_index: int = Field(ge=0)
    image_path: Path
