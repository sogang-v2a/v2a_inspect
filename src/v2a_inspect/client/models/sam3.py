from __future__ import annotations

from pydantic import BaseModel, Field


class Sam3TextPrompt(BaseModel):
    prompt_index: int = Field(ge=0)
    prompt: str = Field(min_length=1)


class Sam3TrackVideoRequest(BaseModel):
    video_id: str
    prompts: list[Sam3TextPrompt]
    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(gt=0)

    def model_post_init(self, __context: object) -> None:
        if self.end_frame_index <= self.start_frame_index:
            raise ValueError("end_frame_index must be greater than start_frame_index.")


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
    prompt_index: int
    points: list[Sam3TrackPoint]
    confidence: float


class Sam3TrackVideoResponse(BaseModel):
    tracks: list[Sam3Track]
