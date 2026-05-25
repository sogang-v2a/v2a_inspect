from __future__ import annotations

import base64

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel

from ..models import LabelScore, LabelScoreRequest, LabelScoreResponse
from ..settings import settings


class Siglip2InferenceClient:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(settings.label_model_id)
        self.model = AutoModel.from_pretrained(
            settings.label_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def score(self, request: LabelScoreRequest) -> LabelScoreResponse:
        label_scores_sum = {label: 0.0 for label in request.labels}
        label_counts = {label: 0 for label in request.labels}

        for image_input in request.images:
            rgb_image = self._decode_image(image_input.image_base64)
            texts = [f"This is a photo of {label}." for label in request.labels]

            inputs = self.processor(
                text=texts,
                images=rgb_image,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.sigmoid(outputs.logits_per_image).cpu().numpy().flatten()

            for idx, label in enumerate(request.labels):
                label_scores_sum[label] += float(probs[idx])
                label_counts[label] += 1

        # Compute average scores
        scores = []
        for label in request.labels:
            if label_counts[label] > 0:
                avg_score = label_scores_sum[label] / label_counts[label]
            else:
                avg_score = 0.0
            scores.append(
                LabelScore(
                    label=label,
                    score=avg_score,
                )
            )

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)

        return LabelScoreResponse(track_id=request.track_id, scores=scores)

    def _decode_image(self, encoded: str) -> np.ndarray:
        if "," in encoded and encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded)
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image payload")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
