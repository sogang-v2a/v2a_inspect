from __future__ import annotations

from typing import List
from ..base import BaseClient
from ..models.embeddings import EmbedRequest, EmbedResponse


class EmbeddingClient(BaseClient):
    """Client for DINOv2 embedding endpoint."""

    async def embed(self, video_id: str, tracks: List[dict]) -> EmbedResponse:
        """
        Get embeddings for tracks in a video.

        Args:
            video_id: The ID of the video (from upload).
            tracks: List of track dictionaries from SAM3 output, each containing:
                   - track_id: str
                   - points: list of dicts with timestamp_seconds, bbox_xyxy, mask_rle, confidence

        Returns:
            EmbedResponse containing embeddings.

        Raises:
            ClientError: If the request fails.
        """
        # Convert tracks to the format expected by EmbedRequest
        track_images = []
        for track in tracks:
            track_images.append({
                "track_id": track["track_id"],
                "points": track["points"]  # Assuming points are already TrackPoint-compatible dicts
            })
        
        request = EmbedRequest(video_id=video_id, tracks=track_images)
        response = await self._request("POST", "/infer/embed", json=request.model_dump())
        return EmbedResponse(**response.json())