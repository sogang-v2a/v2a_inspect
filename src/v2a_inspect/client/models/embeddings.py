from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EncodedImageInput(BaseModel):
    input_id: str
    image_base64: str
    bbox_xyxy: tuple[float, float, float, float] | None = None


class DinoV2ImageInput(EncodedImageInput):
    pass


class DinoV2EmbedImagesRequest(BaseModel):
    inputs: list[DinoV2ImageInput]


class DinoV2Embedding(BaseModel):
    input_id: str
    vector: list[float]
    model_name: str
    embedding_scope: Literal["frame", "region"]


class DinoV2EmbedImagesResponse(BaseModel):
    embeddings: list[DinoV2Embedding]
