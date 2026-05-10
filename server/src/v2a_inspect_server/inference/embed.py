from __future__ import annotations

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel

from ..models import Embedding, EmbedRequest, EmbedResponse
from ..settings import settings


def _box_iou(
    box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return float(inter_area / (box1_area + box2_area - inter_area))


class DinoV2InferenceClient:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Use the embedding model ID from settings
        self.processor = AutoImageProcessor.from_pretrained(settings.embedding_model_id)
        self.model = AutoModel.from_pretrained(
            settings.embedding_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        embeddings: list[Embedding] = []
        for track in request.tracks:
            # Collect image paths for this track's points
            vectors = []
            for point in track.points:
                # Extract the frame at point.timestamp_seconds from the video
                video_exts = [".mp4", ".mov", ".avi", ".mkv"]
                video_path = None
                for ext in video_exts:
                    p = settings.upload_dir / f"{request.video_id}{ext}"
                    if p.exists():
                        video_path = p
                        break
                if not video_path:
                    raise FileNotFoundError(f"Video {request.video_id} not found")

                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                frame_idx = int(point.timestamp_seconds * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    # If we can't read the frame, skip or use zero vector? Let's skip for now.
                    continue
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image with DINOv2
                inputs = self.processor(images=rgb_frame, return_tensors="pt").to(
                    self.device
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # Use the [CLS] token output (first token of the last hidden state)
                embedding_vector = (
                    outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                )
                vectors.append(embedding_vector)

            if vectors:
                # Average the vectors for the track
                avg_vector = np.mean(vectors, axis=0).tolist()
            else:
                avg_vector = [0.0] * self.model.config.hidden_size

            embeddings.append(
                Embedding(
                    track_id=track.track_id,
                    vector=avg_vector,
                    model_name=settings.embedding_model_id,
                )
            )
        return EmbedResponse(embeddings=embeddings)
