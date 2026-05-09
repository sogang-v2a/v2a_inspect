from __future__ import annotations

from pydantic import BaseModel
from .common import ImageRef

class LabelScoreRequest(BaseModel):
    track_id: str | None = None
    labels: list[str]
    images: list[ImageRef]

class LabelScore(BaseModel):
    label: str
    score: float

class LabelScoreResponse(BaseModel):
    track_id: str | None = None
    scores: list[LabelScore]
