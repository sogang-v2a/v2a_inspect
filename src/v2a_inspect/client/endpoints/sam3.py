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
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        score_threshold: float = 0.35,
        min_points: int = 2,
        high_confidence_threshold: float = 0.45,
        match_threshold: float = 0.45,
        batch_size: int | None = None,
    ) -> Sam3TrackVideoResponse:
        if batch_size is not None and batch_size > 0 and len(seeds) > batch_size:
            tracks = []
            for offset in range(0, len(seeds), batch_size):
                chunk = seeds[offset : offset + batch_size]
                response = await self.track_video(
                    video_id,
                    seeds=chunk,
                    start_frame_index=start_frame_index,
                    end_frame_index=end_frame_index,
                    score_threshold=score_threshold,
                    min_points=min_points,
                    high_confidence_threshold=high_confidence_threshold,
                    match_threshold=match_threshold,
                    batch_size=None,
                )
                tracks.extend(
                    track.model_copy(update={"seed_index": track.seed_index + offset})
                    for track in response.tracks
                )
            return Sam3TrackVideoResponse(tracks=tracks)

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
