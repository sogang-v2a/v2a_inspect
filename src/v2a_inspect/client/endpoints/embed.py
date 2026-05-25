from __future__ import annotations

from .base import BaseClient
from ..models.embeddings import (
    DinoV2EmbedImagesRequest,
    DinoV2EmbedImagesResponse,
    DinoV2ImageInput,
)


class EmbeddingClient(BaseClient):
    """Client for DINOv2 frame and region embedding tools."""

    async def embed_images(
        self, inputs: list[DinoV2ImageInput]
    ) -> DinoV2EmbedImagesResponse:
        request = DinoV2EmbedImagesRequest(inputs=inputs)
        response = await self._request(
            "POST", "/infer/dinov2/embed-images", json=request.model_dump()
        )
        return DinoV2EmbedImagesResponse(**response.json())
