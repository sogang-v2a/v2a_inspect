from __future__ import annotations

from pydantic import BaseModel, Field
from .common import ImageRef

class RegionSeed(BaseModel):
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = 1.0
    label_hint: str | None = None

class FrameRef(ImageRef):
    timestamp_seconds: float

class SceneBatch(BaseModel):
    scene_id: str
    frames: list[FrameRef]

class Sam3ExtractRequest(BaseModel):
    scenes: list[SceneBatch]
    prompts_by_scene: dict[str, list[str]] = Field(default_factory=dict)
    seeds_by_scene: dict[str, list[RegionSeed]] = Field(default_factory=dict)
    
    # Inference params
    score_threshold: float = 0.35
    min_points: int = 2
    high_confidence_threshold: float = 0.45
    match_threshold: float = 0.45

class TrackPoint(BaseModel):
    timestamp_seconds: float
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_rle: str | None = None
    confidence: float

class EntityTrack(BaseModel):
    track_id: str
    scene_id: str
    points: list[TrackPoint]
    confidence: float

class Sam3ExtractResponse(BaseModel):
    tracks: list[EntityTrack]
