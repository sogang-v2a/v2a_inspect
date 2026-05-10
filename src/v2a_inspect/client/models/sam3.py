from __future__ import annotations

from pydantic import BaseModel, model_validator
from typing_extensions import Self

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

    @model_validator(mode='after')
    def check_exclusivity(self) -> Self:
        has_spatial = self.bbox_xyxy is not None or (self.points is not None and len(self.points) > 0)
        has_prompt = self.prompt is not None

        if has_spatial and has_prompt:
            raise ValueError("Cannot combine text prompt with bbox/points in the same seed.")
        if not has_spatial and not has_prompt:
            raise ValueError("Must provide either a text prompt, or spatial inputs (bbox/points).")
             
        return self

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