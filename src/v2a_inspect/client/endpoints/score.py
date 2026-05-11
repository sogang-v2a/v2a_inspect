from __future__ import annotations

from .base import BaseClient
from ..models.embeddings import EncodedImageInput
from ..models.labels import LabelScoreRequest, LabelScoreResponse


class ScoringClient(BaseClient):
    """Client for SigLIP2 scoring endpoint."""

    async def score(
        self,
        images: list[EncodedImageInput],
        labels: list[str],
        track_id: str | None = None,
    ) -> LabelScoreResponse:
        """
        Score labels for encoded images.

        Args:
            images: Encoded images to score.
            labels: List of text labels to score against.
            track_id: Optional track ID to include in the response.

        Returns:
            LabelScoreResponse containing scores per label.

        Raises:
            ClientError: If the request fails.
        """
        request = LabelScoreRequest(track_id=track_id, images=images, labels=labels)
        response = await self._request(
            "POST", "/infer/score", json=request.model_dump()
        )
        return LabelScoreResponse(**response.json())
