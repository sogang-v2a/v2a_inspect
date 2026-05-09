from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field


# --- Reused Models (copied from server/src/v2a_inspect_server/models) ---

class PointPrompt(BaseModel):
    x: float
    y: float
    is_positive: bool = True


class VideoSeed(BaseModel):
    timestamp_seconds: float
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    points: Optional[List[PointPrompt]] = None
    prompt: Optional[str] = None
    label_hint: Optional[str] = None


class Sam3ExtractRequest(BaseModel):
    video_id: str
    seeds: List[VideoSeed]
    score_threshold: float = 0.35
    min_points: int = 2
    high_confidence_threshold: float = 0.45
    match_threshold: float = 0.45


class TrackPoint(BaseModel):
    timestamp_seconds: float
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    mask_rle: Optional[str] = None
    confidence: float


class EntityTrack(BaseModel):
    track_id: str
    points: List[TrackPoint]
    confidence: float


class Sam3ExtractResponse(BaseModel):
    tracks: List[EntityTrack]


class TrackImages(BaseModel):
    track_id: str
    points: List[TrackPoint]


class EmbedRequest(BaseModel):
    video_id: str
    tracks: List[TrackImages]


class Embedding(BaseModel):
    track_id: str
    vector: List[float]
    model_name: str


class EmbedResponse(BaseModel):
    embeddings: List[Embedding]


class LabelScore(BaseModel):
    label: str
    score: float


class LabelScoreRequest(BaseModel):
    video_id: str
    track_id: Optional[str] = None
    points: List[TrackPoint]
    labels: List[str]


class LabelScoreResponse(BaseModel):
    track_id: Optional[str] = None
    scores: List[LabelScore]


# --- Client ---


class VideoInferenceClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __del__(self):
        self.client.close()

    def upload_video(self, video_path: str | Path) -> str:
        """Upload a video file and return the video ID.

        Args:
            video_path: Path to the video file (e.g., .mp4, .mov, .avi, .mkv).

        Returns:
            video_id: UUID string for the uploaded video.

        Raises:
            httpx.HTTPStatusError: If the upload fails.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with video_path.open("rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            response = self.client.post(
                f"{self.base_url}/videos/upload",
                files=files,
            )
        response.raise_for_status()
        data = response.json()
        return data["video_id"]

    def infer_sam3(
        self,
        video_id: str,
        seeds: List[VideoSeed],
        score_threshold: float = 0.35,
        min_points: int = 2,
        high_confidence_threshold: float = 0.45,
        match_threshold: float = 0.45,
    ) -> Sam3ExtractResponse:
        """Run SAM3 tracking on the uploaded video.

        Args:
            video_id: ID from upload_video().
            seeds: List of VideoSeed objects (timestamp + bbox/points/prompt).
            score_threshold: Minimum score for a detection to be considered.
            min_points: Minimum number of points required for a track.
            high_confidence_threshold: Threshold for high confidence tracking.
            match_threshold: Threshold for matching detections across frames.

        Returns:
            Sam3ExtractResponse with tracked entities.

        Raises:
            httpx.HTTPStatusError: If the inference fails.
        """
        request = Sam3ExtractRequest(
            video_id=video_id,
            seeds=seeds,
            score_threshold=score_threshold,
            min_points=min_points,
            high_confidence_threshold=high_confidence_threshold,
            match_threshold=match_threshold,
        )
        response = self.client.post(
            f"{self.base_url}/infer/sam3",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return Sam3ExtractResponse(**response.json())

    def embed_video(
        self,
        video_id: str,
        tracks: List[TrackImages],
    ) -> EmbedResponse:
        """Generate DINOv2 embeddings for tracked regions in the video.

        Args:
            video_id: ID from upload_video().
            tracks: List of TrackImages (each with track_id and timestamped points).

        Returns:
            EmbedResponse with embedding vectors per track.

        Raises:
            httpx.HTTPStatusError: If the embedding fails.
        """
        request = EmbedRequest(
            video_id=video_id,
            tracks=tracks,
        )
        response = self.client.post(
            f"{self.base_url}/infer/embed",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return EmbedResponse(**response.json())

    def score_labels(
        self,
        video_id: str,
        track_id: Optional[str],
        points: List[TrackPoint],
        labels: List[str],
    ) -> LabelScoreResponse:
        """Score video regions against text labels using SigLIP2.

        Args:
            video_id: ID from upload_video().
            track_id: Optional track identifier.
            points: List of timestamped points to score.
            labels: List of text labels to score against.

        Returns:
            LabelScoreResponse with scores per label (sorted by confidence).

        Raises:
            httpx.HTTPStatusError: If the scoring fails.
        """
        request = LabelScoreRequest(
            video_id=video_id,
            track_id=track_id,
            points=points,
            labels=labels,
        )
        response = self.client.post(
            f"{self.base_url}/infer/score",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return LabelScoreResponse(**response.json())