from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor

from ..models import (
    Sam3Mask,
    Sam3Seed,
    Sam3SegmentImageRequest,
    Sam3SegmentImageResponse,
    Sam3Track,
    Sam3TrackPoint,
    Sam3TrackVideoRequest,
    Sam3TrackVideoResponse,
)
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


class Sam3InferenceClient:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = Sam3Processor.from_pretrained(settings.sam3_model_id)
        self.model = Sam3Model.from_pretrained(
            settings.sam3_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def track_video(self, request: Sam3TrackVideoRequest) -> Sam3TrackVideoResponse:
        frames = self._load_video_frames(request.video_id)
        tracks: list[Sam3Track] = []

        for seed_index, seed in enumerate(request.seeds):
            points: list[Sam3TrackPoint] = []
            start_index = max(
                0, seed.frame_index if seed.frame_index is not None else 0
            )
            last_bbox = seed.bbox_xyxy

            for frame_index in range(start_index, len(frames)):
                frame = frames[frame_index]
                active_seed = seed
                if frame_index != start_index:
                    if last_bbox is None:
                        break
                    active_seed = Sam3Seed(bbox_xyxy=last_bbox)

                masks = self._segment_image_array(
                    frame,
                    [active_seed],
                    score_threshold=request.score_threshold,
                    max_masks=20,
                )
                best_mask = self._select_tracking_mask(
                    masks,
                    last_bbox,
                    is_first_frame=frame_index == start_index,
                    match_threshold=request.match_threshold,
                )

                if best_mask is None:
                    break
                if best_mask.confidence < request.high_confidence_threshold:
                    break

                points.append(
                    Sam3TrackPoint(
                        frame_index=frame_index,
                        bbox_xyxy=best_mask.bbox_xyxy,
                        mask_rle=best_mask.mask_rle,
                        confidence=best_mask.confidence,
                    )
                )
                last_bbox = best_mask.bbox_xyxy

            if len(points) >= request.min_points:
                confidence = min(point.confidence for point in points)
                tracks.append(
                    Sam3Track(
                        track_id=f"seed-{seed_index}",
                        seed_index=seed_index,
                        points=points,
                        confidence=confidence,
                    )
                )

        return Sam3TrackVideoResponse(tracks=tracks)

    def segment_image(
        self, request: Sam3SegmentImageRequest
    ) -> Sam3SegmentImageResponse:
        image = self._load_request_image(
            image_path=request.image_path,
            video_id=request.video_id,
            frame_index=request.frame_index,
        )
        masks = self._segment_image_array(
            image,
            request.seeds,
            score_threshold=request.score_threshold,
            max_masks=request.max_masks,
        )
        return Sam3SegmentImageResponse(masks=masks)

    def _find_video_path(self, video_id: str) -> Path:
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            path = settings.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video {video_id} not found")

    def _load_video_frames(self, video_id: str) -> list[np.ndarray]:
        video_path = self._find_video_path(video_id)
        cap = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise ValueError(f"No frames could be loaded from video {video_id}")
        return frames

    def _load_request_image(
        self,
        *,
        image_path: str | None,
        video_id: str | None,
        frame_index: int | None,
    ) -> np.ndarray:
        if image_path is not None:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image {image_path} not found")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if video_id is None or frame_index is None:
            raise ValueError("video_id and frame_index are required for video frames")
        return self._load_video_frame(video_id, frame_index)

    def _load_video_frame(self, video_id: str, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError("frame_index must be greater than or equal to 0")
        video_path = self._find_video_path(video_id)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(
                f"Could not read frame {frame_index} from video {video_id}"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _segment_image_array(
        self,
        image: np.ndarray,
        seeds: list[Sam3Seed],
        *,
        score_threshold: float,
        max_masks: int,
    ) -> list[Sam3Mask]:
        masks: list[Sam3Mask] = []

        for seed_index, seed in enumerate(seeds):
            inputs = self._build_processor_inputs(image, seed)
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            processed = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                mask_threshold=score_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            boxes = processed.get("boxes", [])
            scores = processed.get("scores", [])
            for result_index, (box, score) in enumerate(zip(boxes, scores)):
                if len(masks) >= max_masks:
                    return masks
                masks.append(
                    Sam3Mask(
                        mask_id=f"seed-{seed_index}-mask-{result_index}",
                        bbox_xyxy=tuple(float(value) for value in box.tolist()),
                        confidence=float(score),
                        source_seed_index=seed_index,
                    )
                )

        masks.sort(key=lambda mask: mask.confidence, reverse=True)
        return masks[:max_masks]

    def _build_processor_inputs(self, image: np.ndarray, seed: Sam3Seed) -> dict:
        if seed.prompt is not None:
            return self.processor(images=image, text=seed.prompt, return_tensors="pt")

        input_points = None
        input_labels = None
        if seed.points:
            input_points = [[[point.x, point.y] for point in seed.points]]
            input_labels = [[[1 if point.is_positive else 0 for point in seed.points]]]

        if input_points is not None and seed.bbox_xyxy is not None:
            return self.processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=[[[list(seed.bbox_xyxy)]]],
                return_tensors="pt",
            )
        if input_points is not None:
            return self.processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
        if seed.bbox_xyxy is not None:
            return self.processor(
                images=image,
                input_boxes=[[[list(seed.bbox_xyxy)]]],
                return_tensors="pt",
            )

        raise ValueError("Seed must include prompt, bbox, or points")

    def _select_tracking_mask(
        self,
        masks: list[Sam3Mask],
        last_bbox: tuple[float, float, float, float] | None,
        *,
        is_first_frame: bool,
        match_threshold: float,
    ) -> Sam3Mask | None:
        if not masks:
            return None
        if is_first_frame or last_bbox is None:
            return max(masks, key=lambda mask: mask.confidence)

        best_mask = None
        best_score = 0.0
        for mask in masks:
            combined_score = (_box_iou(last_bbox, mask.bbox_xyxy) * 0.7) + (
                mask.confidence * 0.3
            )
            if combined_score > best_score and combined_score > match_threshold:
                best_score = combined_score
                best_mask = mask.model_copy(update={"confidence": combined_score})
        return best_mask
