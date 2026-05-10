from __future__ import annotations

from pydantic import BaseModel
from .sam3 import TrackPoint


class TrackImages(BaseModel):
    track_id: str
    points: list[TrackPoint]


class EmbedRequest(BaseModel):
    video_id: str
    tracks: list[TrackImages]


class Embedding(BaseModel):
    track_id: str
    vector: list[float]
    model_name: str


class EmbedResponse(BaseModel):
    embeddings: list[Embedding]