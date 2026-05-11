from __future__ import annotations

from pydantic import BaseModel

from .embeddings import EncodedImageInput


class LabelScoreRequest(BaseModel):
    track_id: str | None = None
    images: list[EncodedImageInput]
    labels: list[str]


class LabelScore(BaseModel):
    label: str
    score: float


class LabelScoreResponse(BaseModel):
    track_id: str | None = None
    scores: list[LabelScore]
