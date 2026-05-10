from __future__ import annotations

from pydantic import BaseModel


class VideoIDResponse(BaseModel):
    video_id: str