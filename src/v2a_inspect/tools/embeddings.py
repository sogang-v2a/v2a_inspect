from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from .remote import post_json
from .types import CanonicalLabel, EntityEmbedding, LabelScore


class EmbeddingRequest(BaseModel):
    items: list[dict[str, object]] = Field(default_factory=list)


class EmbeddingRunpodClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        model_name: str = "dinov2",
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
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
        response = post_json(
            self.endpoint_url,
            {"input": payload},
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        output = response.get("output", response)
        if not isinstance(output, dict):
            raise TypeError("Embedding endpoint returned an invalid payload.")
        embeddings = output.get("embeddings", [])
        return [
            EntityEmbedding(
                track_id=item["track_id"],
                model_name=item.get("model_name", self.model_name),
                vector=item.get("vector", []),
            )
            for item in embeddings
            if isinstance(item, dict) and "track_id" in item
        ]


class Siglip2LabelClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

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
        response = post_json(
            self.endpoint_url,
            {"input": payload},
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        output = response.get("output", response)
        if not isinstance(output, dict):
            raise TypeError("Label endpoint returned an invalid payload.")
        scores = [
            LabelScore(label=item["label"], score=item["score"])
            for item in output.get("scores", [])
            if isinstance(item, dict) and "label" in item and "score" in item
        ]
        best_label = max(scores, key=lambda item: item.score).label if scores else ""
        return CanonicalLabel(group_id=group_id, label=best_label, scores=scores)


def _read_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
