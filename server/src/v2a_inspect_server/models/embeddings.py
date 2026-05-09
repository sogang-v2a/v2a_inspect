from __future__ import annotations

from pydantic import BaseModel
from .common import ImageRef

class TrackImages(BaseModel):
    track_id: str
    images: list[ImageRef]

class EmbedRequest(BaseModel):
    tracks: list[TrackImages]

class Embedding(BaseModel):
    track_id: str
    vector: list[float]
    model_name: str

class EmbedResponse(BaseModel):
    embeddings: list[Embedding]
