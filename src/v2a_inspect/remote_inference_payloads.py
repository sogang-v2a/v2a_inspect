from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from v2a_inspect.tools.types import Sam3RegionSeed


class UploadedImageRef(BaseModel):
    upload_key: str = Field(min_length=1)
    filename: str | None = None


class UploadedFrameRef(UploadedImageRef):
    scene_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)


class UploadedFrameBatch(BaseModel):
    scene_index: int = Field(ge=0)
    frames: list[UploadedFrameRef] = Field(default_factory=list)


class Sam3ExtractManifest(BaseModel):
    frame_batches: list[UploadedFrameBatch] = Field(default_factory=list)
    prompts_by_scene: dict[int, list[str]] | None = None
    region_seeds_by_scene: dict[int, list[Sam3RegionSeed]] | None = None
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    min_points: int = Field(default=2, ge=1)
    high_confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    match_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class TrackImageBatch(BaseModel):
    track_id: str = Field(min_length=1)
    images: list[UploadedImageRef] = Field(default_factory=list)


class EmbedImagesManifest(BaseModel):
    tracks: list[TrackImageBatch] = Field(default_factory=list)


class LabelScoreManifest(BaseModel):
    track_id: str | None = None
    labels: list[str] = Field(default_factory=list)
    images: list[UploadedImageRef] = Field(default_factory=list)


class RemoteInferenceJsonMixin:
    def model_json_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
