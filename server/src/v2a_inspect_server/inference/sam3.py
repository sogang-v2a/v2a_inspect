from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..settings import settings

if settings.opencv_video_backend.lower() == "ffmpeg":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_FFMPEG", "100000")
if settings.opencv_ffmpeg_capture_options:
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        settings.opencv_ffmpeg_capture_options,
    )

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor
from torchvision.ops import masks_to_boxes

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


logger = logging.getLogger(__name__)


class Sam3InferenceClient:
    def __init__(self) -> None:
        self.predictor = build_sam3_multiplex_video_predictor(
            max_num_objects=settings.sam31_max_num_objects,
            use_fa3=settings.sam31_use_fa3,
            use_rope_real=settings.sam31_use_rope_real,
            compile=settings.sam31_compile,
            warm_up=settings.sam31_warm_up,
        )
        self._lock = threading.Lock()

    def close(self) -> None:
        shutdown = getattr(self.predictor, "shutdown", None)
        if shutdown is not None:
            shutdown()

    def track_video(self, request: Sam3TrackVideoRequest) -> Sam3TrackVideoResponse:
        request_started_at = time.perf_counter()
        video_path = self._find_video_path(request.video_id)

        if (
            request.start_frame_index is not None
            and request.end_frame_index is not None
        ):
            decode_started_at = time.perf_counter()
            frames, width, height = self._read_video_frame_range(
                video_path,
                start_frame_index=request.start_frame_index,
                end_frame_index=request.end_frame_index,
            )
            logger.info(
                "SAM3 decoded frame range video_id=%s range=[%s,%s) frames=%s size=%sx%s elapsed=%.3fs",
                request.video_id,
                request.start_frame_index,
                request.end_frame_index,
                len(frames),
                width,
                height,
                time.perf_counter() - decode_started_at,
            )
            try:
                tracks = self._track_resource(
                    resource_path=frames,
                    seeds=self._localize_seed_frame_indexes(
                        request.seeds,
                        start_frame_index=request.start_frame_index,
                    ),
                    width=width,
                    height=height,
                    output_prob_thresh=request.score_threshold,
                    frame_index_offset=request.start_frame_index,
                )
            finally:
                self._close_frames(frames)
            logger.info(
                "SAM3 completed frame-range tracking video_id=%s range=[%s,%s) tracks=%s elapsed=%.3fs",
                request.video_id,
                request.start_frame_index,
                request.end_frame_index,
                len(tracks),
                time.perf_counter() - request_started_at,
            )
            return Sam3TrackVideoResponse(tracks=tracks)

        size_started_at = time.perf_counter()
        width, height = self._read_video_size(video_path)
        logger.info(
            "SAM3 read video size video_id=%s size=%sx%s elapsed=%.3fs",
            request.video_id,
            width,
            height,
            time.perf_counter() - size_started_at,
        )
        tracks = self._track_resource(
            resource_path=video_path,
            seeds=request.seeds,
            width=width,
            height=height,
            output_prob_thresh=request.score_threshold,
            frame_index_offset=0,
        )
        logger.info(
            "SAM3 completed full-video tracking video_id=%s tracks=%s elapsed=%.3fs",
            request.video_id,
            len(tracks),
            time.perf_counter() - request_started_at,
        )

        return Sam3TrackVideoResponse(tracks=tracks)

    def _track_resource(
        self,
        *,
        resource_path: Path | list[Image.Image],
        seeds: list[Sam3Seed],
        width: int,
        height: int,
        output_prob_thresh: float,
        frame_index_offset: int,
    ) -> list[Sam3Track]:
        with self._lock:
            session_started_at = time.perf_counter()
            session_id = self._start_session(resource_path)
            logger.info(
                "SAM3 started tracking session session_id=%s resource=%s elapsed=%.3fs",
                session_id,
                _resource_description(resource_path),
                time.perf_counter() - session_started_at,
            )
            try:
                for seed_index, seed in enumerate(seeds):
                    prompt_started_at = time.perf_counter()
                    self._add_seed_prompt(
                        session_id=session_id,
                        seed=seed,
                        seed_index=seed_index,
                        width=width,
                        height=height,
                        output_prob_thresh=output_prob_thresh,
                    )
                    logger.info(
                        "SAM3 added seed prompt session_id=%s seed_index=%s seed_type=%s elapsed=%.3fs",
                        session_id,
                        seed_index,
                        _seed_type(seed),
                        time.perf_counter() - prompt_started_at,
                    )
                propagate_started_at = time.perf_counter()
                tracks = self._propagate_tracks(
                    session_id,
                    width=width,
                    height=height,
                    output_prob_thresh=output_prob_thresh,
                    frame_index_offset=frame_index_offset,
                )
                logger.info(
                    "SAM3 propagated tracks session_id=%s tracks=%s elapsed=%.3fs",
                    session_id,
                    len(tracks),
                    time.perf_counter() - propagate_started_at,
                )
            finally:
                self._close_session(session_id)

        return tracks

    def segment_image(
        self, request: Sam3SegmentImageRequest
    ) -> Sam3SegmentImageResponse:
        if request.video_id is not None and request.frame_index is not None:
            video_path = self._find_video_path(request.video_id)
            with tempfile.TemporaryDirectory() as temp_dir:
                frame_dir = Path(temp_dir)
                width, height = self._write_video_frame_to_directory(
                    video_path,
                    frame_index=request.frame_index,
                    frame_dir=frame_dir,
                )
                masks = self._segment_resource(
                    resource_path=frame_dir,
                    frame_index=0,
                    seeds=request.seeds,
                    width=width,
                    height=height,
                    max_masks=request.max_masks,
                    output_prob_thresh=request.score_threshold,
                )
            return Sam3SegmentImageResponse(masks=masks)

        if request.image_path is None:
            raise ValueError("image_path is required for image segmentation")

        image_path = Path(request.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image {image_path} not found")
        width, height = self._read_image_size(image_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            frame_dir = Path(temp_dir)
            frame_path = frame_dir / "00000.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image {image_path} not found")
            cv2.imwrite(str(frame_path), image)
            masks = self._segment_resource(
                resource_path=frame_dir,
                frame_index=0,
                seeds=request.seeds,
                width=width,
                height=height,
                max_masks=request.max_masks,
                output_prob_thresh=request.score_threshold,
            )

        return Sam3SegmentImageResponse(masks=masks)

    def _segment_resource(
        self,
        *,
        resource_path: Path,
        frame_index: int,
        seeds: list[Sam3Seed],
        width: int,
        height: int,
        max_masks: int,
        output_prob_thresh: float,
    ) -> list[Sam3Mask]:
        with self._lock:
            session_id = self._start_session(resource_path)
            try:
                outputs = None
                for seed_index, seed in enumerate(seeds):
                    outputs = self._add_seed_prompt(
                        session_id=session_id,
                        seed=seed,
                        seed_index=seed_index,
                        width=width,
                        height=height,
                        default_frame_index=frame_index,
                        output_prob_thresh=output_prob_thresh,
                    )
                if outputs is None:
                    return []
                masks = self._outputs_to_masks(
                    outputs,
                    width=width,
                    height=height,
                    max_masks=max_masks,
                )
            finally:
                self._close_session(session_id)

        return masks

    def _start_session(self, resource_path: Path | list[Image.Image]) -> str:
        if isinstance(resource_path, Path):
            sam3_resource_path: str | list[Image.Image] = str(resource_path)
        else:
            sam3_resource_path = resource_path

        init_kwargs: dict[str, Any] = {
            "resource_path": sam3_resource_path,
            "offload_video_to_cpu": False,
        }
        if hasattr(self.predictor, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.predictor.async_loading_frames
        if hasattr(self.predictor, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.predictor.video_loader_type

        # The current SAM3.1 request dispatcher passes offload_state_to_cpu to
        # the multiplex initializer, but that initializer no longer accepts it.
        inference_state = self.predictor.model.init_state(**init_kwargs)
        session_id = str(uuid.uuid4())
        self.predictor._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return session_id

    def _close_session(self, session_id: str) -> None:
        self.predictor.handle_request(
            request={"type": "close_session", "session_id": session_id}
        )

    def _add_seed_prompt(
        self,
        *,
        session_id: str,
        seed: Sam3Seed,
        seed_index: int,
        width: int,
        height: int,
        default_frame_index: int | None = None,
        output_prob_thresh: float | None = None,
    ) -> dict[str, Any]:
        frame_index = seed.frame_index
        if frame_index is None:
            frame_index = 0 if default_frame_index is None else default_frame_index

        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
        }
        if output_prob_thresh is not None:
            request["output_prob_thresh"] = output_prob_thresh

        if seed.prompt is not None:
            request["text"] = seed.prompt
        elif seed.points:
            request["points"] = self._relative_points(seed, width=width, height=height)
            request["point_labels"] = [
                1 if point.is_positive else 0 for point in seed.points
            ]
            request["obj_id"] = seed_index + 1
        elif seed.bbox_xyxy is not None:
            request["bounding_boxes"] = [
                self._bbox_xyxy_to_relative_xywh(
                    seed.bbox_xyxy,
                    width=width,
                    height=height,
                )
            ]
            request["bounding_box_labels"] = [1]
            request["obj_id"] = seed_index + 1
        else:
            raise ValueError("Seed must include prompt, bbox, or points")

        response = self.predictor.handle_request(request=request)
        return dict(response["outputs"])

    def _propagate_tracks(
        self,
        session_id: str,
        *,
        width: int,
        height: int,
        output_prob_thresh: float,
        frame_index_offset: int,
    ) -> list[Sam3Track]:
        points_by_obj_id: dict[int, list[Sam3TrackPoint]] = {}

        for response in self.predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "output_prob_thresh": output_prob_thresh,
            }
        ):
            frame_index = int(response["frame_index"]) + frame_index_offset
            outputs = response["outputs"]
            for obj_id, bbox_xyxy, confidence, mask_rle in self._iter_output_objects(
                outputs,
                width=width,
                height=height,
            ):
                points_by_obj_id.setdefault(obj_id, []).append(
                    Sam3TrackPoint(
                        frame_index=frame_index,
                        bbox_xyxy=bbox_xyxy,
                        mask_rle=mask_rle,
                        confidence=confidence,
                    )
                )

        tracks = []
        for obj_id, points in sorted(points_by_obj_id.items()):
            if not points:
                continue
            confidence = min(point.confidence for point in points)
            tracks.append(
                Sam3Track(
                    track_id=str(obj_id),
                    seed_index=obj_id,
                    points=points,
                    confidence=confidence,
                )
            )
        return tracks

    def _localize_seed_frame_indexes(
        self,
        seeds: list[Sam3Seed],
        *,
        start_frame_index: int,
    ) -> list[Sam3Seed]:
        localized_seeds = []
        for seed in seeds:
            if seed.frame_index is None:
                localized_seeds.append(seed)
                continue
            localized_seeds.append(
                seed.model_copy(
                    update={"frame_index": seed.frame_index - start_frame_index}
                )
            )
        return localized_seeds

    def _outputs_to_masks(
        self,
        outputs: dict[str, Any],
        *,
        width: int,
        height: int,
        max_masks: int,
    ) -> list[Sam3Mask]:
        masks = []
        for obj_id, bbox_xyxy, confidence, mask_rle in self._iter_output_objects(
            outputs,
            width=width,
            height=height,
        ):
            masks.append(
                Sam3Mask(
                    mask_id=str(obj_id),
                    bbox_xyxy=bbox_xyxy,
                    mask_rle=mask_rle,
                    confidence=confidence,
                    source_seed_index=obj_id,
                )
            )
            if len(masks) >= max_masks:
                break
        return masks

    def _iter_output_objects(
        self, outputs: dict[str, Any], *, width: int, height: int
    ) -> list[tuple[int, tuple[float, float, float, float], float, str | None]]:
        obj_ids = self._to_python_list(outputs.get("out_obj_ids", []))
        boxes_xywh = self._to_python_list(outputs.get("out_boxes_xywh", []))
        probabilities = self._to_python_list(outputs.get("out_probs", []))
        binary_masks = outputs.get("out_binary_masks")

        objects = []
        for index, obj_id in enumerate(obj_ids):
            bbox_xyxy = None
            mask_rle = None
            if index < len(boxes_xywh):
                bbox_xyxy = self._relative_xywh_to_absolute_xyxy(
                    boxes_xywh[index],
                    width=width,
                    height=height,
                )
            elif binary_masks is not None:
                bbox_xyxy = self._mask_to_bbox_xyxy(binary_masks[index])

            if binary_masks is not None:
                mask_tensor = binary_masks[index]
                if not isinstance(mask_tensor, torch.Tensor):
                    mask_tensor = torch.as_tensor(np.asarray(mask_tensor))
                mask_array = mask_tensor.detach().cpu().bool().numpy().squeeze()
                if mask_array.ndim == 2:
                    encoded_mask = mask_utils.encode(
                        np.asfortranarray(mask_array.astype(np.uint8))
                    )
                    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                    mask_rle = json.dumps(
                        {
                            "encoding": "coco_rle",
                            "size": encoded_mask["size"],
                            "counts": encoded_mask["counts"],
                        },
                        separators=(",", ":"),
                    )

            if bbox_xyxy is None:
                continue

            confidence = 1.0
            if index < len(probabilities):
                confidence = float(probabilities[index])

            objects.append((int(obj_id), bbox_xyxy, confidence, mask_rle))

        return objects

    def _find_video_path(self, video_id: str) -> Path:
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            path = settings.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video {video_id} not found")

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

    def _read_video_size(self, video_path: Path) -> tuple[int, int]:
        cap = self._open_video_capture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not read video dimensions from {video_path}")
        return width, height

    def _read_image_size(self, image_path: Path) -> tuple[int, int]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image {image_path} not found")
        height, width = image.shape[:2]
        return width, height

    def _write_video_frame_to_directory(
        self, video_path: Path, *, frame_index: int, frame_dir: Path
    ) -> tuple[int, int]:
        if frame_index < 0:
            raise ValueError("frame_index must be greater than or equal to 0")
        cap = self._open_video_capture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame {frame_index} from {video_path}")
        frame_path = frame_dir / "00000.jpg"
        cv2.imwrite(str(frame_path), frame)
        height, width = frame.shape[:2]
        return width, height

    def _read_video_frame_range(
        self,
        video_path: Path,
        *,
        start_frame_index: int,
        end_frame_index: int,
    ) -> tuple[list[Image.Image], int, int]:
        if start_frame_index < 0:
            raise ValueError("start_frame_index must be greater than or equal to 0")
        if end_frame_index <= start_frame_index:
            raise ValueError("end_frame_index must be greater than start_frame_index")

        cap = self._open_video_capture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        try:
            frames = []
            width = 0
            height = 0
            frame_index = start_frame_index
            while frame_index < end_frame_index:
                ret, frame = cap.read()
                if not ret:
                    break

                if not frames:
                    height, width = frame.shape[:2]

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
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not read video dimensions from {video_path}")
        return frames, width, height

    def _close_frames(self, frames: list[Image.Image]) -> None:
        for frame in frames:
            frame.close()
        frames.clear()

    def _relative_points(
        self, seed: Sam3Seed, *, width: int, height: int
    ) -> list[list[float]]:
        points = []
        if seed.points is None:
            return points
        for point in seed.points:
            points.append([point.x / width, point.y / height])
        return points

    def _bbox_xyxy_to_relative_xywh(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        *,
        width: int,
        height: int,
    ) -> list[float]:
        x1, y1, x2, y2 = bbox_xyxy
        return [
            x1 / width,
            y1 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]

    def _relative_xywh_to_absolute_xyxy(
        self,
        bbox_xywh: list[float],
        *,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        x, y, box_width, box_height = bbox_xywh
        x1 = x * width
        y1 = y * height
        x2 = (x + box_width) * width
        y2 = (y + box_height) * height
        return (float(x1), float(y1), float(x2), float(y2))

    def _mask_to_bbox_xyxy(self, mask: Any) -> tuple[float, float, float, float] | None:
        tensor = mask
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(np.asarray(mask))
        tensor = tensor.detach().cpu().bool()
        if not tensor.any():
            return None
        box = masks_to_boxes(tensor.unsqueeze(0))[0].tolist()
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    def _to_python_list(self, value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if value is None:
            return []
        return list(value)


def _resource_description(resource_path: Path | list[Image.Image]) -> str:
    if isinstance(resource_path, Path):
        return str(resource_path)
    return f"{len(resource_path)} in-memory frames"


def _seed_type(seed: Sam3Seed) -> str:
    if seed.prompt is not None:
        return "text"
    if seed.points:
        return "points"
    if seed.bbox_xyxy is not None:
        return "bbox"
    return "unknown"
