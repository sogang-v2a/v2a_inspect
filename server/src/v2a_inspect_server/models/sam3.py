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


class Sam3Mask(BaseModel):
    mask_id: str
    bbox_xyxy: tuple[float, float, float, float]
    mask_rle: str | None = Field(
        default=None,
        description="Compressed COCO RLE JSON payload.",
    )
    confidence: float
    source_seed_index: int | None = None


class Sam3SegmentFrameItem(BaseModel):
    request_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    seed: Sam3Seed
    max_masks: int = Field(default=5, ge=1)


class Sam3SegmentFramesRequest(BaseModel):
    video_id: str
    items: list[Sam3SegmentFrameItem]
    score_threshold: float = 0.35
    batch_size: int = Field(default=32, ge=1)


class Sam3SegmentFrameResult(BaseModel):
    request_index: int
    frame_index: int
    masks: list[Sam3Mask] = Field(default_factory=list)


class Sam3SegmentFrameError(BaseModel):
    request_index: int
    frame_index: int
    message: str


class Sam3SegmentFramesResponse(BaseModel):
    results: list[Sam3SegmentFrameResult] = Field(default_factory=list)
    errors: list[Sam3SegmentFrameError] = Field(default_factory=list)


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
