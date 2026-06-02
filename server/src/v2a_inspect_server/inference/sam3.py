from __future__ import annotations

import logging
import os
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from ..models import (
    Sam3TextPrompt,
    Sam3Track,
    Sam3TrackPoint,
    Sam3TrackVideoRequest,
    Sam3TrackVideoResponse,
)
from ..settings import settings

if settings.opencv_video_backend.lower() == "ffmpeg":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_FFMPEG", "100000")
if settings.opencv_ffmpeg_capture_options:
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        settings.opencv_ffmpeg_capture_options,
    )

logger = logging.getLogger("uvicorn.error")


class Sam3InferenceClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.attention_implementation = ""
        self.model = self._load_model()
        self.processor = Sam3VideoProcessor.from_pretrained(
            settings.sam3_model_id,
            size={
                "height": settings.sam3_image_size,
                "width": settings.sam3_image_size,
            },
        )
        self.device = _model_device(self.model)
        self.dtype = _torch_dtype(settings.sam3_dtype)
        logger.info(
            "Loaded HF SAM3 video model model_id=%s image_size=%s dtype=%s attention=%s device=%s",
            settings.sam3_model_id,
            settings.sam3_image_size,
            settings.sam3_dtype,
            self.attention_implementation,
            self.device,
        )

    def close(self) -> None:
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def track_video(self, request: Sam3TrackVideoRequest) -> Sam3TrackVideoResponse:
        request_started_at = time.perf_counter()
        if not request.prompts:
            return Sam3TrackVideoResponse(tracks=[])

        video_path = self._find_video_path(request.video_id)
        frames = self._read_video_frame_range(
            video_path,
            start_frame_index=request.start_frame_index,
            end_frame_index=request.end_frame_index,
        )
        logger.info(
            "SAM3 loaded video range video_id=%s range=[%s,%s) frames=%s",
            request.video_id,
            request.start_frame_index,
            request.end_frame_index,
            len(frames),
        )

        with self._lock:
            tracks = self._track_frames(
                frames,
                prompts=request.prompts,
                frame_index_offset=request.start_frame_index,
            )

        logger.info(
            "SAM3 completed HF tracking video_id=%s range=[%s,%s) prompts=%s tracks=%s elapsed=%.3fs",
            request.video_id,
            request.start_frame_index,
            request.end_frame_index,
            len(request.prompts),
            len(tracks),
            time.perf_counter() - request_started_at,
        )
        return Sam3TrackVideoResponse(tracks=tracks)

    def _load_model(self) -> Sam3VideoModel:
        attempts = [settings.sam3_attention_implementation]
        if settings.sam3_attention_implementation != "sdpa":
            attempts.append("sdpa")

        last_error: Exception | None = None
        for attention_implementation in attempts:
            try:
                config = Sam3VideoConfig.from_pretrained(settings.sam3_model_id)
                config.image_size = settings.sam3_image_size
                config.score_threshold_detection = (
                    settings.sam3_score_threshold_detection
                )
                config.new_det_thresh = settings.sam3_new_det_thresh
                model = Sam3VideoModel.from_pretrained(
                    settings.sam3_model_id,
                    config=config,
                    device_map="auto",
                    dtype=_torch_dtype(settings.sam3_dtype),
                    attn_implementation=attention_implementation,
                )
                model.eval()
                self.attention_implementation = attention_implementation
                return model
            except Exception as exc:
                last_error = exc
                logger.exception(
                    "Failed to load SAM3 with attention=%s",
                    attention_implementation,
                )
        assert last_error is not None
        raise last_error

    def _track_frames(
        self,
        frames: list[Image.Image],
        *,
        prompts: list[Sam3TextPrompt],
        frame_index_offset: int,
    ) -> list[Sam3Track]:
        prompt_texts = [prompt.prompt for prompt in prompts]
        prompt_entries_by_text: dict[str, list[Sam3TextPrompt]] = defaultdict(list)
        for prompt in prompts:
            prompt_entries_by_text[prompt.prompt].append(prompt)
        duplicate_prompts = [
            prompt
            for prompt, entries in prompt_entries_by_text.items()
            if len(entries) > 1
        ]
        if duplicate_prompts:
            logger.warning(
                "SAM3 prompt texts are duplicated; tracks will be fanned out prompts=%s",
                duplicate_prompts,
            )

        inference_session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=prompt_texts,
        )

        points_by_track_key: dict[tuple[int, int], list[Sam3TrackPoint]] = {}
        object_ids_by_prompt_index: dict[int, set[int]] = defaultdict(set)
        max_frame_num_to_track = max(0, len(frames) - 1)
        with torch.inference_mode():
            for model_outputs in self.model.propagate_in_video_iterator(
                inference_session=inference_session,
                start_frame_idx=0,
                max_frame_num_to_track=max_frame_num_to_track,
                show_progress_bar=False,
            ):
                processed = self.processor.postprocess_outputs(
                    inference_session,
                    model_outputs,
                )
                self._append_processed_frame(
                    processed,
                    frame_index=int(model_outputs.frame_idx) + frame_index_offset,
                    prompt_entries_by_text=prompt_entries_by_text,
                    object_ids_by_prompt_index=object_ids_by_prompt_index,
                    points_by_track_key=points_by_track_key,
                )

        tracks = []
        for (prompt_index, object_id), points in sorted(points_by_track_key.items()):
            if not points:
                continue
            confidence = min(point.confidence for point in points)
            tracks.append(
                Sam3Track(
                    track_id=f"{prompt_index}:{object_id}",
                    prompt_index=prompt_index,
                    points=points,
                    confidence=confidence,
                )
            )
        return tracks

    def _append_processed_frame(
        self,
        processed: dict[str, Any],
        *,
        frame_index: int,
        prompt_entries_by_text: dict[str, list[Sam3TextPrompt]],
        object_ids_by_prompt_index: dict[int, set[int]],
        points_by_track_key: dict[tuple[int, int], list[Sam3TrackPoint]],
    ) -> None:
        object_ids = _to_int_list(processed.get("object_ids"))
        scores = _to_float_list(processed.get("scores"))
        boxes = _to_nested_float_list(processed.get("boxes"))
        masks = processed.get("masks")

        object_index_by_id = {
            object_id: index for index, object_id in enumerate(object_ids)
        }
        prompt_to_obj_ids = processed.get("prompt_to_obj_ids") or {}
        for prompt_text, object_ids_for_prompt in prompt_to_obj_ids.items():
            prompt_entries = prompt_entries_by_text.get(prompt_text, [])
            if not prompt_entries:
                continue
            for object_id in _to_int_list(object_ids_for_prompt):
                object_index = object_index_by_id.get(object_id)
                if object_index is None:
                    continue
                for prompt in prompt_entries:
                    object_ids_by_prompt_index[prompt.prompt_index].add(object_id)
                    point = self._track_point_from_processed_object(
                        frame_index=frame_index,
                        object_index=object_index,
                        scores=scores,
                        boxes=boxes,
                        masks=masks,
                    )
                    if point is not None:
                        points_by_track_key.setdefault(
                            (prompt.prompt_index, object_id),
                            [],
                        ).append(point)

    def _track_point_from_processed_object(
        self,
        *,
        frame_index: int,
        object_index: int,
        scores: list[float],
        boxes: list[list[float]],
        masks: Any,
    ) -> Sam3TrackPoint | None:
        bbox_xyxy = None
        if object_index < len(boxes) and len(boxes[object_index]) >= 4:
            box = boxes[object_index]
            bbox_xyxy = (
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            )
        mask_rle = _mask_to_rle(masks, object_index)
        if bbox_xyxy is None and mask_rle is None:
            return None

        confidence = 1.0
        if object_index < len(scores):
            confidence = scores[object_index]
        return Sam3TrackPoint(
            frame_index=frame_index,
            bbox_xyxy=bbox_xyxy,
            mask_rle=mask_rle,
            confidence=confidence,
        )

    def _find_video_path(self, video_id: str) -> Path:
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            path = settings.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video {video_id} not found")

    def _read_video_frame_range(
        self,
        video_path: Path,
        *,
        start_frame_index: int,
        end_frame_index: int,
    ) -> list[Image.Image]:
        cap = self._open_video_capture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        try:
            frames = []
            frame_index = start_frame_index
            while frame_index < end_frame_index:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                frame_index += 1
        finally:
            cap.release()

        expected_frame_count = end_frame_index - start_frame_index
        if len(frames) != expected_frame_count:
            raise ValueError(
                "Could not read requested frame range "
                f"[{start_frame_index}, {end_frame_index}) from {video_path}; "
                f"read {len(frames)} of {expected_frame_count} frames"
            )
        return frames

    def _open_video_capture(self, video_path: Path) -> cv2.VideoCapture:
        backend = self._opencv_video_backend()
        if settings.opencv_ffmpeg_capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                settings.opencv_ffmpeg_capture_options
            )

        if settings.opencv_hw_acceleration:
            params = [
                cv2.CAP_PROP_HW_ACCELERATION,
                cv2.VIDEO_ACCELERATION_ANY,
            ]
            if settings.opencv_hw_device is not None:
                params.extend([cv2.CAP_PROP_HW_DEVICE, settings.opencv_hw_device])
            cap = cv2.VideoCapture(str(video_path), backend, params)
            if cap.isOpened():
                return cap
            cap.release()

        cap = cv2.VideoCapture(str(video_path), backend)
        if cap.isOpened():
            return cap
        cap.release()
        raise ValueError(f"Could not open video with OpenCV: {video_path}")

    def _opencv_video_backend(self) -> int:
        backend = settings.opencv_video_backend.lower()
        if backend != "ffmpeg":
            raise ValueError(f"Unsupported OpenCV video backend: {backend}")
        return cv2.CAP_FFMPEG


def _torch_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported SAM3 dtype: {value}")


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [int(item) for item in value]


def _to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [float(item) for item in value]


def _to_nested_float_list(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [[float(item) for item in row] for row in value]


def _mask_to_rle(masks: Any, object_index: int) -> str | None:
    if masks is None:
        return None
    if isinstance(masks, torch.Tensor):
        if object_index >= masks.shape[0]:
            return None
        mask = masks[object_index].detach().cpu().bool().numpy()
    else:
        mask_array = np.asarray(masks)
        if object_index >= mask_array.shape[0]:
            return None
        mask = mask_array[object_index].astype(bool)
    mask = np.squeeze(mask)
    if mask.ndim != 2 or not mask.any():
        return None
    encoded_mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
    return json.dumps(
        {
            "encoding": "coco_rle",
            "size": encoded_mask["size"],
            "counts": encoded_mask["counts"],
        },
        separators=(",", ":"),
    )
