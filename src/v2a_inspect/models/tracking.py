from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, computed_field, model_validator
from typing_extensions import Self

from .base import SchemaModel
from .initial_analysis import ObjectSeed


class MaskRef(SchemaModel):
    """
    Reference to a segmentation mask for one frame.

    The mask can be stored either:
    - inline as an RLE string
    - externally as a file path
    """

    mask_id: UUID = Field(default_factory=uuid4)

    encoding: Literal["rle"] = "rle"
    rle: str | None = None
    path: Path | None = None

    @model_validator(mode="after")
    def check_mask_storage(self) -> Self:
        has_rle = self.rle is not None
        has_path = self.path is not None

        if has_rle == has_path:
            raise ValueError("Provide exactly one mask storage: rle or path.")

        return self


class SceneTrackPoint(SchemaModel):
    """
    Tracking result for one frame.

    Assumptions:
    - frame_index is based on the prepared 1280x720, 30fps, audio-free video.
    - bbox_xyxy uses pixel coordinates of the prepared video.
    """

    frame_index: int = Field(ge=0)

    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask: MaskRef | None = None

    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_visual_output(self) -> Self:
        if self.bbox_xyxy is None and self.mask is None:
            raise ValueError("SceneTrackPoint must have bbox_xyxy or mask.")

        return self


class SceneTrack(SchemaModel):
    """
    SAM3 tracking result inside one InitialScene.

    A SceneTrack:
    - is owned by InitialScene
    - is usually produced from one ObjectSeed
    - contains tracking results for every frame in its visible interval
    - is not a global object identity
    """

    scene_track_id: UUID = Field(default_factory=uuid4)

    source_object_seed: ObjectSeed | None = None
    tracking_prompt: str

    points: list[SceneTrackPoint] = Field(default_factory=list)

    confidence: float = Field(ge=0, le=1)
    notes: str | None = None

    @model_validator(mode="after")
    def check_points(self) -> Self:
        if not self.points:
            raise ValueError("SceneTrack must contain at least one point.")

        return self

    @computed_field
    @property
    def start_frame_index(self) -> int:
        return min(point.frame_index for point in self.points)

    @computed_field
    @property
    def end_frame_index(self) -> int:
        return max(point.frame_index for point in self.points) + 1

    @computed_field
    @property
    def frame_count(self) -> int:
        return self.end_frame_index - self.start_frame_index
