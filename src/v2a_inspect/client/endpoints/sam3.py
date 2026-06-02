from __future__ import annotations

from .base import BaseClient
from ..models.sam3 import (
    Sam3TextPrompt,
    Sam3TrackVideoRequest,
    Sam3TrackVideoResponse,
)


class SAM3Client(BaseClient):
    """Client for SAM3 video text-prompt tracking."""

    async def track_video(
        self,
        video_id: str,
        prompts: list[Sam3TextPrompt],
        start_frame_index: int,
        end_frame_index: int,
    ) -> Sam3TrackVideoResponse:
        request = Sam3TrackVideoRequest(
            video_id=video_id,
            prompts=prompts,
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
        )
        response = await self._request(
            "POST", "/infer/sam3/track-video", json=request.model_dump()
        )
        return Sam3TrackVideoResponse(**response.json())
