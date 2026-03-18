from pydantic import BaseModel, Field
from typing import Optional, List


class TimeRange(BaseModel):
    start: float = Field(
        description="Start time in seconds with 0.1s precision (e.g., 1.3, 4.7, 12.1)"
    )
    end: float = Field(
        description="End time in seconds with 0.1s precision (e.g., 2.8, 6.4, 15.3)"
    )


class SceneObject(BaseModel):
    description: str = Field(
        description="Object description as a short text clip. Must include specific count (e.g. '2 people playing guitars', '1 dog running')"
    )
    time_range: TimeRange = Field(
        description="Time range when this object appears. Must be within the scene's time range."
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Track group ID assigned post-analysis for temporal consistency",
    )
    canonical_description: Optional[str] = Field(
        default=None, description="Unified canonical description for the group"
    )


class Scene(BaseModel):
    scene_index: int = Field(description="0-based scene index")
    time_range: TimeRange = Field(description="Time range of this scene")
    background_sound: str = Field(
        description="Short text clip describing appropriate background sound/music for this scene. Include extra objects beyond the main 2 as part of background sound."
    )
    objects: List[SceneObject] = Field(
        description="Main objects in the scene. Maximum 2. Each must include specific counts. If more than 2 notable objects exist, describe extras in background_sound.",
        max_length=2,
    )
    background_group_id: Optional[str] = Field(
        default=None, description="Group ID for background track assigned post-analysis"
    )
    background_canonical: Optional[str] = Field(
        default=None,
        description="Unified canonical description for the background group",
    )


class VideoSceneAnalysis(BaseModel):
    total_duration: float = Field(description="Total video duration in seconds")
    scenes: List[Scene] = Field(description="List of scenes detected in the video")
