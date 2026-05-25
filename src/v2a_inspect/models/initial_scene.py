from uuid import UUID, uuid4

from pydantic import Field, computed_field

from .base import SchemaModel
from .initial_analysis import InitialSceneAnalysis
from .keyframe import Keyframe
from .tracking import SceneTrack


class InitialScene(SchemaModel):
    """
    Coarse scene segment produced by PySceneDetect.

    This is an early mechanical split, not the final semantic Scene used for export.

    Frame interval convention:
    - [start_frame_index, end_frame_index)
    - start is inclusive
    - end is exclusive
    """

    initial_scene_id: UUID = Field(default_factory=uuid4)

    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)

    keyframes: list[Keyframe] = Field(default_factory=list)
    initial_analysis: InitialSceneAnalysis | None = None
    scene_tracks: list[SceneTrack] = Field(default_factory=list)

    @computed_field
    @property
    def frame_count(self) -> int:
        return self.end_frame_index - self.start_frame_index
