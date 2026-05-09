from __future__ import annotations

from pydantic import BaseModel, Field

class ImageRef(BaseModel):
    """Maps a metadata entry to an actual file uploaded in the multipart request"""
    upload_id: str = Field(..., description="Key matching the file in the multipart upload")
