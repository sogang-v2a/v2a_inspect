from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class PointPrompt(BaseModel):
    x: float
    y: float
    is_positive: bool = True


class Sam3Seed(BaseModel):
    frame_index: int | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    points: list[PointPrompt] | None = None
    prompt: str | None = None
    label_hint: str | None = None

    @model_validator(mode="after")
    def check_prompt(self) -> Self:
        has_spatial = self.bbox_xyxy is not None or bool(self.points)
        has_prompt = self.prompt is not None

        if has_spatial and has_prompt:
            raise ValueError("Cannot combine text prompt with bbox/points in one seed.")
        if not has_spatial and not has_prompt:
            raise ValueError("Must provide either prompt, bbox, or points.")
        return self


class Sam3TrackVideoRequest(BaseModel):
    video_id: str
    seeds: list[Sam3Seed]
    start_frame_index: int | None = Field(default=None, ge=0)
    end_frame_index: int | None = Field(default=None, gt=0)
    score_threshold: float = 0.35
    min_points: int = 2
    high_confidence_threshold: float = 0.45
    match_threshold: float = 0.45

    @model_validator(mode="after")
    def check_frame_range(self) -> Self:
        has_start = self.start_frame_index is not None
        has_end = self.end_frame_index is not None
        if has_start != has_end:
            raise ValueError(
                "Provide both start_frame_index and end_frame_index, or neither."
            )
        if self.start_frame_index is None or self.end_frame_index is None:
            return self
        if self.end_frame_index <= self.start_frame_index:
            raise ValueError("end_frame_index must be greater than start_frame_index.")
        for seed in self.seeds:
            if seed.frame_index is None:
                continue
            if not self.start_frame_index <= seed.frame_index < self.end_frame_index:
                raise ValueError(
                    "Seed frame_index must be inside "
                    "[start_frame_index, end_frame_index)."
                )
        return self


class Sam3SegmentImageRequest(BaseModel):
    image_path: str | None = None
    video_id: str | None = None
    frame_index: int | None = None
    seeds: list[Sam3Seed]
    score_threshold: float = 0.35
    max_masks: int = 5

    @model_validator(mode="after")
    def check_image_source(self) -> Self:
        has_image_path = self.image_path is not None
        has_video_frame = self.video_id is not None and self.frame_index is not None

        if has_image_path == has_video_frame:
            raise ValueError(
                "Provide exactly one image source: image_path or video_id with frame_index."
            )
        return self


class Sam3Mask(BaseModel):
    mask_id: str
    bbox_xyxy: tuple[float, float, float, float]
    mask_rle: str | None = Field(
        default=None,
        description="Compressed COCO RLE JSON payload.",
    )
    confidence: float
    source_seed_index: int | None = None


class Sam3TrackPoint(BaseModel):
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_rle: str | None = Field(
        default=None,
        description="Compressed COCO RLE JSON payload.",
    )
    confidence: float


class Sam3Track(BaseModel):
    track_id: str
    seed_index: int
    points: list[Sam3TrackPoint]
    confidence: float


class Sam3TrackVideoResponse(BaseModel):
    tracks: list[Sam3Track]


class Sam3SegmentImageResponse(BaseModel):
    masks: list[Sam3Mask]
