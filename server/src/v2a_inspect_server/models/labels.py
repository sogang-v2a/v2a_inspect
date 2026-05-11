from __future__ import annotations

from pydantic import BaseModel
from .sam3 import Sam3TrackPoint


class LabelScoreRequest(BaseModel):
    video_id: str
    track_id: str | None = None
    points: list[Sam3TrackPoint]
    labels: list[str]


class LabelScore(BaseModel):
    label: str
    score: float


class LabelScoreResponse(BaseModel):
    track_id: str | None = None
    scores: list[LabelScore]
