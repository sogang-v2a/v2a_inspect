from __future__ import annotations

from .base import BaseClient
from ..models.sam3 import (
    PointPrompt,
    Sam3Seed,
    Sam3SegmentFrameItem,
    Sam3SegmentFramesRequest,
    Sam3SegmentFramesResponse,
    Sam3TrackVideoRequest,
    Sam3TrackVideoResponse,
)


class SAM3Client(BaseClient):
    """Client for SAM3 tracking and image segmentation tools."""

    async def track_video(
        self,
        video_id: str,
        seeds: list[Sam3Seed],
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        score_threshold: float = 0.35,
        min_points: int = 2,
        high_confidence_threshold: float = 0.45,
        match_threshold: float = 0.45,
    ) -> Sam3TrackVideoResponse:
        request = Sam3TrackVideoRequest(
            video_id=video_id,
            seeds=seeds,
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            score_threshold=score_threshold,
            min_points=min_points,
            high_confidence_threshold=high_confidence_threshold,
            match_threshold=match_threshold,
        )
        response = await self._request(
            "POST", "/infer/sam3/track-video", json=request.model_dump()
        )
        return Sam3TrackVideoResponse(**response.json())

    async def segment_frames(
        self,
        video_id: str,
        items: list[Sam3SegmentFrameItem],
        score_threshold: float = 0.35,
        batch_size: int = 32,
    ) -> Sam3SegmentFramesResponse:
        request = Sam3SegmentFramesRequest(
            video_id=video_id,
            items=items,
            score_threshold=score_threshold,
            batch_size=batch_size,
        )
        response = await self._request(
            "POST", "/infer/sam3/segment-frames", json=request.model_dump()
        )
        return Sam3SegmentFramesResponse(**response.json())

    @staticmethod
    def seed_from_prompt(prompt: str, frame_index: int | None = None) -> Sam3Seed:
        return Sam3Seed(frame_index=frame_index, prompt=prompt)

    @staticmethod
    def seed_from_points(
        points: list[tuple[float, float, bool]], frame_index: int | None = None
    ) -> Sam3Seed:
        return Sam3Seed(
            frame_index=frame_index,
            points=[
                PointPrompt(x=x, y=y, is_positive=is_positive)
                for x, y, is_positive in points
            ],
        )
