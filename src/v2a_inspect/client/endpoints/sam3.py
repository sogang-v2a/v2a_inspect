from __future__ import annotations

from typing import List, Optional

from .base import BaseClient
from ..models.sam3 import (
    PointPrompt,
    VideoSeed,
    Sam3ExtractRequest,
    Sam3ExtractResponse,
)


class SAM3Client(BaseClient):
    """Client for SAM3 tracking endpoint."""

    async def track(
        self,
        video_id: str,
        prompt: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        points: Optional[List[List[float]]] = None,
        score_threshold: float = 0.35,
        min_points: int = 2,
        high_confidence_threshold: float = 0.45,
        match_threshold: float = 0.45,
    ) -> Sam3ExtractResponse:
        """
        Run SAM3 tracking on a video.

        Args:
            video_id: The ID of the video (from upload).
            prompt: Text prompt for tracking (mutually exclusive with bbox/points).
            bbox: Bounding box [x1, y1, x2, y2] (mutually exclusive with prompt).
            points: List of points, each [x, y, is_positive] where is_positive is bool.
            score_threshold: Minimum score to consider a detection.
            min_points: Minimum number of points for a prompt.
            high_confidence_threshold: Threshold for high confidence matches.
            match_threshold: Threshold for matching tracks between frames.

        Returns:
            Sam3ExtractResponse containing tracks.

        Raises:
            ClientError: If the request fails.
            ValueError: If mutual exclusivity constraints are violated.
        """
        # Prepare seeds
        seeds: List[VideoSeed] = []
        if prompt is not None:
            seeds.append(VideoSeed(timestamp_seconds=0.0, prompt=prompt))
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError("bbox must be a list of 4 numbers [x1, y1, x2, y2]")
            seeds.append(VideoSeed(timestamp_seconds=0.0, bbox_xyxy=tuple(bbox)))
        if points is not None:
            point_objs = []
            for p in points:
                if len(p) != 3:
                    raise ValueError("Each point must be [x, y, is_positive]")
                point_objs.append(
                    PointPrompt(x=p[0], y=p[1], is_positive=bool(p[2]))
                )
            seeds.append(VideoSeed(timestamp_seconds=0.0, points=point_objs))

        # The VideoSeed model enforces mutual exclusivity, so we can rely on that.
        request = Sam3ExtractRequest(
            video_id=video_id,
            seeds=seeds,
            score_threshold=score_threshold,
            min_points=min_points,
            high_confidence_threshold=high_confidence_threshold,
            match_threshold=match_threshold,
        )

        response = await self._request(
            "POST", "/infer/sam3", json=request.model_dump()
        )
        return Sam3ExtractResponse(**response.json())