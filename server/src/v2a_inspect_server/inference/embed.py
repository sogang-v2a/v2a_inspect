from __future__ import annotations

import base64

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from ..models import (
    DinoV2Embedding,
    DinoV2EmbedImagesRequest,
    DinoV2EmbedImagesResponse,
    EncodedImageInput,
)
from ..settings import settings


class DinoV2InferenceClient:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(settings.embedding_model_id)
        self.model = AutoModel.from_pretrained(
            settings.embedding_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def embed_images(
        self, request: DinoV2EmbedImagesRequest
    ) -> DinoV2EmbedImagesResponse:
        embeddings = []
        for image_input in request.inputs:
            image = self._decode_image(image_input)
            embedding_scope = "frame"
            if image_input.bbox_xyxy is not None:
                image = self._crop_image(image, image_input.bbox_xyxy)
                embedding_scope = "region"

            embeddings.append(
                DinoV2Embedding(
                    input_id=image_input.input_id,
                    vector=self._embed_image(image),
                    model_name=settings.embedding_model_id,
                    embedding_scope=embedding_scope,
                )
            )
        return DinoV2EmbedImagesResponse(embeddings=embeddings)

    def _decode_image(self, image_input: EncodedImageInput) -> np.ndarray:
        image_bytes = self._decode_base64(image_input.image_base64)
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image {image_input.input_id}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _decode_base64(self, encoded: str) -> bytes:
        if "," in encoded and encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        return base64.b64decode(encoded)

    def _crop_image(
        self, image: np.ndarray, bbox_xyxy: tuple[float, float, float, float]
    ) -> np.ndarray:
        height, width = image.shape[:2]
        x1 = max(0, min(width, int(round(bbox_xyxy[0]))))
        y1 = max(0, min(height, int(round(bbox_xyxy[1]))))
        x2 = max(0, min(width, int(round(bbox_xyxy[2]))))
        y2 = max(0, min(height, int(round(bbox_xyxy[3]))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox for crop: {bbox_xyxy}")
        return image[y1:y2, x1:x2]

    def _embed_image(self, image: np.ndarray) -> list[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().tolist()
