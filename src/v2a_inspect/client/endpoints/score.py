from __future__ import annotations

from typing import List, Optional
from .base import BaseClient
from ..models.labels import LabelScoreRequest, LabelScoreResponse
from ..models.sam3 import TrackPoint


class ScoringClient(BaseClient):
    """Client for SigLIP2 scoring endpoint."""

    async def score(
        self,
        video_id: str,
        labels: List[str],
        track_id: Optional[str] = None,
        points: Optional[List[TrackPoint]] = None,
    ) -> LabelScoreResponse:
        """
        Score labels for a video or specific track.

        Args:
            video_id: The ID of the video (from upload).
            labels: List of text labels to score against.
            track_id: Optional track ID to score (if None, scores whole video).
            points: Optional list of points (if track_id is None, points for whole video).
                   Each point should be a dict with keys: timestamp_seconds, bbox_xyxy (optional), mask_rle (optional), confidence.
                   Note: The server's TrackPoint model has: timestamp_seconds, bbox_xyxy (tuple[float, float, float, float] | None), mask_rle (str | None), confidence (float)

        Returns:
            LabelScoreResponse containing scores per label.

        Raises:
            ClientError: If the request fails.
        """
        request = LabelScoreRequest(
            video_id=video_id, track_id=track_id, points=points or [], labels=labels
        )
        response = await self._request(
            "POST", "/infer/score", json=request.model_dump()
        )
        return LabelScoreResponse(**response.json())
