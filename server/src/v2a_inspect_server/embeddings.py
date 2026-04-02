from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from v2a_inspect.tools.types import CanonicalLabel, EntityEmbedding, LabelScore

from .execution import execute_service
from .providers import GpuProvider, ProviderServiceConfig


class EmbeddingRequest(BaseModel):
    items: list[dict[str, object]] = Field(default_factory=list)


class EmbeddingClient:
    def __init__(
        self,
        *,
        provider: GpuProvider,
        service: ProviderServiceConfig,
        model_name: str = "dinov2",
    ) -> None:
        self.provider = provider
        self.service = service
        self.model_name = model_name

    def embed_images(
        self, image_paths_by_track: dict[str, list[str]]
    ) -> list[EntityEmbedding]:
        payload = EmbeddingRequest(
            items=[
                {
                    "track_id": track_id,
                    "images_base64": [_read_base64(path) for path in image_paths],
                }
                for track_id, image_paths in image_paths_by_track.items()
            ]
        ).model_dump(mode="json")
        result = execute_service(self.provider, self.service, payload)
        embeddings = result.payload.get("embeddings", [])
        return [
            EntityEmbedding(
                track_id=item["track_id"],
                model_name=item.get("model_name", self.model_name),
                vector=item.get("vector", []),
            )
            for item in embeddings
            if isinstance(item, dict) and "track_id" in item
        ]


class LabelClient:
    def __init__(
        self,
        *,
        provider: GpuProvider,
        service: ProviderServiceConfig,
    ) -> None:
        self.provider = provider
        self.service = service

    def score_labels(
        self,
        *,
        group_id: str,
        image_paths: list[str],
        labels: list[str],
    ) -> CanonicalLabel:
        payload = {
            "images_base64": [_read_base64(path) for path in image_paths],
            "labels": labels,
        }
        result = execute_service(self.provider, self.service, payload)
        scores = [
            LabelScore(label=item["label"], score=item["score"])
            for item in result.payload.get("scores", [])
            if isinstance(item, dict) and "label" in item and "score" in item
        ]
        best_label = max(scores, key=lambda item: item.score).label if scores else ""
        return CanonicalLabel(group_id=group_id, label=best_label, scores=scores)


EmbeddingRunpodClient = EmbeddingClient
Siglip2LabelClient = LabelClient


def _read_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
