from __future__ import annotations

from collections.abc import Sequence

from v2a_inspect.tools.types import CanonicalLabel, EntityEmbedding, LabelScore

from .image_features import image_embedding_vector, summarize_image_paths


class EmbeddingClient:
    def __init__(self, *, model_name: str = "native-visual-embedding") -> None:
        self.model_name = model_name

    def embed_images(
        self, image_paths_by_track: dict[str, list[str]]
    ) -> list[EntityEmbedding]:
        return [
            EntityEmbedding(
                track_id=track_id,
                model_name=self.model_name,
                vector=image_embedding_vector(image_paths),
            )
            for track_id, image_paths in image_paths_by_track.items()
            if image_paths
        ]


class LabelClient:
    def score_labels(
        self,
        *,
        group_id: str,
        image_paths: list[str],
        labels: list[str],
    ) -> CanonicalLabel:
        stats = summarize_image_paths(image_paths)
        candidate_scores = {
            "background": _clamp01(
                (1.0 - stats.edge_density) * 0.55
                + (1.0 - stats.contrast) * 0.30
                + (1.0 - stats.colorfulness) * 0.15
            ),
            "object": _clamp01(
                stats.edge_density * 0.40
                + stats.contrast * 0.35
                + stats.colorfulness * 0.25
            ),
            "person": _clamp01(
                stats.vertical_energy * 0.40
                + stats.edge_density * 0.30
                + stats.colorfulness * 0.15
                + stats.brightness * 0.15
            ),
            "vehicle": _clamp01(
                stats.horizontal_energy * 0.35
                + stats.edge_density * 0.30
                + stats.contrast * 0.20
                + (1.0 - stats.colorfulness) * 0.15
            ),
            "animal": _clamp01(
                stats.colorfulness * 0.35
                + stats.edge_density * 0.30
                + stats.contrast * 0.20
                + stats.brightness * 0.15
            ),
        }
        normalized_labels = _normalize_labels(labels)
        scores = [
            LabelScore(label=label, score=candidate_scores.get(label, 0.25))
            for label in normalized_labels
        ]
        scores.sort(key=lambda item: item.score, reverse=True)
        best_label = scores[0].label if scores else ""
        return CanonicalLabel(group_id=group_id, label=best_label, scores=scores)


EmbeddingRunpodClient = EmbeddingClient
Siglip2LabelClient = LabelClient


def _normalize_labels(labels: Sequence[str]) -> list[str]:
    normalized = [label.strip().lower() for label in labels if label.strip()]
    return normalized or ["object"]


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
