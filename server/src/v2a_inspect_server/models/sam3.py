from __future__ import annotations

from pydantic import BaseModel

class PointPrompt(BaseModel):
    x: float
    y: float
    is_positive: bool = True

class VideoSeed(BaseModel):
    timestamp_seconds: float
    bbox_xyxy: tuple[float, float, float, float] | None = None
    points: list[PointPrompt] | None = None
    prompt: str | None = None
    label_hint: str | None = None

class Sam3ExtractRequest(BaseModel):
    video_id: str
    seeds: list[VideoSeed]
    
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
    points: list[TrackPoint]
    confidence: float

class Sam3ExtractResponse(BaseModel):
    tracks: list[EntityTrack]
