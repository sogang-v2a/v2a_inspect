from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from ..models import (
    DinoV2Embedding,
    DinoV2EmbedImagesRequest,
    DinoV2EmbedImagesResponse,
    DinoV2ImageInput,
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
            image = self._load_image_input(image_input)
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

    def _find_video_path(self, video_id: str) -> Path:
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            path = settings.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video {video_id} not found")

    def _load_image_input(self, image_input: DinoV2ImageInput) -> np.ndarray:
        if image_input.image_path is not None:
            image = cv2.imread(image_input.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image {image_input.image_path} not found")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image_input.video_id is None or image_input.timestamp_seconds is None:
            raise ValueError(
                "video_id and timestamp_seconds are required for video frames"
            )
        return self._load_video_frame(
            image_input.video_id, image_input.timestamp_seconds
        )

    def _load_video_frame(self, video_id: str, timestamp_seconds: float) -> np.ndarray:
        video_path = self._find_video_path(video_id)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_seconds * fps))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(
                f"Could not read frame at {timestamp_seconds} seconds from video {video_id}"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
