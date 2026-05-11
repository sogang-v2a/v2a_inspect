from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator
from typing_extensions import Self


class DinoV2ImageInput(BaseModel):
    input_id: str
    image_path: str | None = None
    video_id: str | None = None
    timestamp_seconds: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def check_image_source(self) -> Self:
        has_image_path = self.image_path is not None
        has_video_frame = (
            self.video_id is not None and self.timestamp_seconds is not None
        )

        if has_image_path == has_video_frame:
            raise ValueError(
                "Provide exactly one image source: image_path or video_id with timestamp_seconds."
            )
        return self


class DinoV2EmbedImagesRequest(BaseModel):
    inputs: list[DinoV2ImageInput]


class DinoV2Embedding(BaseModel):
    input_id: str
    vector: list[float]
    model_name: str
    embedding_scope: Literal["frame", "region"]


class DinoV2EmbedImagesResponse(BaseModel):
    embeddings: list[DinoV2Embedding]
