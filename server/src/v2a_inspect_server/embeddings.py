from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from v2a_inspect.tools.types import CanonicalLabel, EntityEmbedding, LabelScore

from .model_runtime import (
    inference_device,
    inference_dtype,
    load_rgb_images,
    move_inputs_to_device,
)


class EmbeddingClient:
    def __init__(self, *, model_dir: str | Path, model_name: str = "facebook/dinov2-base") -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.device = inference_device()
        self.dtype = inference_dtype()
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def embed_images(
        self, image_paths_by_track: dict[str, list[str]]
    ) -> list[EntityEmbedding]:
        embeddings: list[EntityEmbedding] = []
        for track_id, image_paths in image_paths_by_track.items():
            if not image_paths:
                continue
            images = load_rgb_images(image_paths)
            inputs = self.processor(images=images, return_tensors="pt")
            batch_inputs = move_inputs_to_device(dict(inputs), self.device)
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
            last_hidden_state = outputs[0]
            cls_tokens = last_hidden_state[:, 0, :]
            vector = cls_tokens.mean(dim=0).float().cpu().tolist()
            embeddings.append(
                EntityEmbedding(
                    track_id=track_id,
                    model_name=self.model_name,
                    vector=vector,
                )
            )
        return embeddings


class LabelClient:
    def __init__(self, *, model_dir: str | Path, model_name: str = "google/siglip2-base-patch16-224") -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.device = inference_device()
        self.dtype = inference_dtype()
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def score_image_labels(
        self,
        *,
        image_paths: list[str],
        labels: list[str],
    ) -> list[LabelScore]:
        normalized_labels = _normalize_labels(labels)
        images = load_rgb_images(image_paths)
        texts = [f"This is a photo of {label}." for label in normalized_labels]
        inputs = self.processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        batch_inputs = move_inputs_to_device(dict(inputs), self.device)
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
        logits = outputs.logits_per_image
        probabilities = torch.sigmoid(logits).mean(dim=0).float().cpu().tolist()
        scores = [
            LabelScore(label=label, score=float(probability))
            for label, probability in zip(normalized_labels, probabilities, strict=False)
        ]
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores

    def score_labels(
        self,
        *,
        group_id: str,
        image_paths: list[str],
        labels: list[str],
    ) -> CanonicalLabel:
        scores = self.score_image_labels(image_paths=image_paths, labels=labels)
        best_label = scores[0].label if scores else ""
        return CanonicalLabel(group_id=group_id, label=best_label, scores=scores)


def _normalize_labels(labels: Sequence[str]) -> list[str]:
    normalized = [label.strip().lower() for label in labels if label.strip()]
    return normalized or ["object"]
