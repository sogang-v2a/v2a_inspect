from __future__ import annotations

from .base import BaseClient
from ..models.sam3 import (
    PointPrompt,
    Sam3Seed,
    Sam3SegmentImageRequest,
    Sam3SegmentImageResponse,
    Sam3TrackVideoRequest,
    Sam3TrackVideoResponse,
)


class SAM3Client(BaseClient):
    """Client for SAM3 tracking and image segmentation tools."""

    async def track_video(
        self,
        video_id: str,
        seeds: list[Sam3Seed],
        score_threshold: float = 0.35,
        min_points: int = 2,
        high_confidence_threshold: float = 0.45,
        match_threshold: float = 0.45,
    ) -> Sam3TrackVideoResponse:
        request = Sam3TrackVideoRequest(
            video_id=video_id,
            seeds=seeds,
            score_threshold=score_threshold,
            min_points=min_points,
            high_confidence_threshold=high_confidence_threshold,
            match_threshold=match_threshold,
        )
        response = await self._request(
            "POST", "/infer/sam3/track-video", json=request.model_dump()
        )
        return Sam3TrackVideoResponse(**response.json())

    async def segment_image(
        self,
        seeds: list[Sam3Seed],
        image_path: str | None = None,
        video_id: str | None = None,
        frame_index: int | None = None,
        score_threshold: float = 0.35,
        max_masks: int = 5,
    ) -> Sam3SegmentImageResponse:
        request = Sam3SegmentImageRequest(
            image_path=image_path,
            video_id=video_id,
            frame_index=frame_index,
            seeds=seeds,
            score_threshold=score_threshold,
            max_masks=max_masks,
        )
        response = await self._request(
            "POST", "/infer/sam3/segment-image", json=request.model_dump()
        )
        return Sam3SegmentImageResponse(**response.json())

    @staticmethod
    def seed_from_bbox(
        bbox: tuple[float, float, float, float], frame_index: int | None = None
    ) -> Sam3Seed:
        return Sam3Seed(frame_index=frame_index, bbox_xyxy=bbox)

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
