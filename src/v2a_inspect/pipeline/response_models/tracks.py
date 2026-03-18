import re
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict

from .scenes import VideoSceneAnalysis


class ModelSelection(BaseModel):
    """TTA/VTA model selection result for a single track or group."""

    reasoning: str
    model_type: Literal["TTA", "VTA"]
    confidence: float = Field(..., ge=0.0, le=1.0)  # 0.0–1.0
    vta_score: float  # combined VTA preference (video motion + event coupling)
    tta_score: float  # combined TTA preference (source diversity + object count bias)
    rule_based: bool = (
        False  # True = deterministic rule (background, etc.), False = LLM judgment
    )


class RawTrack(BaseModel):
    """One track extracted from a Scene (background or object)."""

    track_id: str  # e.g. "s0_bg", "s0_obj0", "s1_obj1"
    scene_index: int
    kind: Literal["background", "object"]
    description: str
    start: float
    end: float
    obj_index: Optional[int] = None  # None for backgrounds
    n_scene_objects: int = 0  # number of object tracks in the same scene
    model_selection: Optional[ModelSelection] = None  # assigned post-grouping

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def validate_track_id(cls, track_id: str) -> str:
        pattern = r"^s\d+_(bg|obj\d+)$"
        if not re.match(pattern, track_id):
            raise ValueError(
                f"Invalid track_id format: '{track_id}'. Expected format 's{{scene_index}}_{{bg|obj{{obj_index}}}}', e.g. 's0_bg', 's1_obj0'."
            )
        return track_id


class TrackGroup(BaseModel):
    """A set of RawTracks that represent the same real-world audio entity."""

    group_id: str
    canonical_description: str  # description used for audio generation
    member_ids: List[str]  # track_ids belonging to this group
    vlm_verified: bool = False
    model_selection: Optional[ModelSelection] = None  # group-level representative


class GroupedAnalysis(BaseModel):
    """VideoSceneAnalysis annotated with group assignments."""

    scene_analysis: VideoSceneAnalysis  # VideoSceneAnalysis (annotated copy)
    raw_tracks: List[RawTrack]
    groups: List[TrackGroup]
    track_to_group: Dict[str, str]  # track_id -> group_id
