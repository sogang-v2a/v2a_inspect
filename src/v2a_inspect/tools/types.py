from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VideoProbe(BaseModel):
    video_path: str
    duration_seconds: float = Field(ge=0.0)
    fps: float | None = Field(default=None, gt=0.0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    frame_count: int | None = Field(default=None, ge=0)
    codec_name: str | None = None
    format_name: str | None = None
    has_audio_stream: bool | None = None


class SceneBoundary(BaseModel):
    scene_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    strategy: Literal["fixed_window", "ffmpeg_scene_detect"] = "fixed_window"


class SampledFrame(BaseModel):
    scene_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    image_path: str


class FrameBatch(BaseModel):
    scene_index: int = Field(ge=0)
    frames: list[SampledFrame] = Field(default_factory=list)


class Sam3VisualFeatures(BaseModel):
    motion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    interaction_score: float = Field(default=0.0, ge=0.0, le=1.0)
    crowd_score: float = Field(default=0.0, ge=0.0, le=1.0)
    camera_dynamics_score: float = Field(default=0.0, ge=0.0, le=1.0)
    trajectory_motion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    scale_change_score: float = Field(default=0.0, ge=0.0, le=1.0)
    continuity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overlap_score: float = Field(default=0.0, ge=0.0, le=1.0)
    scene_crowd_score: float = Field(default=0.0, ge=0.0, le=1.0)
    scene_camera_dynamics_score: float = Field(default=0.0, ge=0.0, le=1.0)


class Sam3TrackPoint(BaseModel):
    timestamp_seconds: float = Field(ge=0.0)
    frame_path: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox_xyxy: list[float] | None = Field(default=None, min_length=4, max_length=4)
    mask_rle: str | None = None


class Sam3EntityTrack(BaseModel):
    track_id: str
    scene_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    label_hint: str | None = None
    points: list[Sam3TrackPoint] = Field(default_factory=list)
    features: Sam3VisualFeatures = Field(default_factory=Sam3VisualFeatures)


class Sam3TrackSet(BaseModel):
    provider: str = "sam3"
    strategy: Literal["prompt_free", "prompt_seeded", "text_recovery"] = "prompt_free"
    tracks: list[Sam3EntityTrack] = Field(default_factory=list)


class EntityEmbedding(BaseModel):
    track_id: str
    model_name: str = "dinov2"
    vector: list[float] = Field(default_factory=list)


class CandidateGroup(BaseModel):
    group_id: str
    member_track_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CandidateGroupSet(BaseModel):
    groups: list[CandidateGroup] = Field(default_factory=list)


class LabelScore(BaseModel):
    label: str
    score: float = Field(ge=0.0)


class CanonicalLabel(BaseModel):
    group_id: str
    label: str
    scores: list[LabelScore] = Field(default_factory=list)


class TrackRoutingDecision(BaseModel):
    track_id: str
    model_type: Literal["TTA", "VTA"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class GroupRoutingDecision(BaseModel):
    group_id: str
    model_type: Literal["TTA", "VTA"]
    confidence: float = Field(ge=0.0, le=1.0)
    member_track_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""
    aggregate_method: Literal["majority_vote_geometric_mean"] = (
        "majority_vote_geometric_mean"
    )
