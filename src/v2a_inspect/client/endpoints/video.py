from __future__ import annotations

import os

from .base import BaseClient
from ..models.video import VideoIDResponse


class VideoClient(BaseClient):
    """Client for video upload endpoint."""

    async def upload(self, video_path: str) -> VideoIDResponse:
        """
        Upload a video file to the server.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoIDResponse containing the video_id.

        Raises:
            ClientError: If the upload fails.
            FileNotFoundError: If the video file does not exist.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Prepare the file for upload
        filename = os.path.basename(video_path)
        with open(video_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}
            response = await self._request("POST", "/videos/upload", files=files)

        # Parse the response
        return VideoIDResponse(**response.json())
