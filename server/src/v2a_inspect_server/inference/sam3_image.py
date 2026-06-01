from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import sam3
import torch
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as Sam3Image,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

from ..models import (
    Sam3Mask,
    Sam3Seed,
    Sam3SegmentFrameError,
    Sam3SegmentFrameItem,
    Sam3SegmentFrameResult,
    Sam3SegmentFramesRequest,
    Sam3SegmentFramesResponse,
)
from ..settings import settings

if settings.opencv_video_backend.lower() == "ffmpeg":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_FFMPEG", "100000")
if settings.opencv_ffmpeg_capture_options:
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        settings.opencv_ffmpeg_capture_options,
    )


class Sam3ImageInferenceClient:
    def __init__(self) -> None:
        bpe_path = (
            Path(sam3.__file__).resolve().parent.parent
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        )
        self.model = build_sam3_image_model(bpe_path=str(bpe_path))
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=1008,
                    max_size=1008,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self._query_counter = 1
        self._lock = threading.Lock()

    def segment_frames(
        self, request: Sam3SegmentFramesRequest
    ) -> Sam3SegmentFramesResponse:
        video_path = self._find_video_path(request.video_id)
        frames = self._read_video_frames(
            video_path,
            frame_indexes={item.frame_index for item in request.items},
        )
        results: list[Sam3SegmentFrameResult] = []
        errors: list[Sam3SegmentFrameError] = []

        items_by_frame: dict[int, list[Sam3SegmentFrameItem]] = {}
        for item in request.items:
            if item.frame_index not in frames:
                errors.append(
                    Sam3SegmentFrameError(
                        request_index=item.request_index,
                        frame_index=item.frame_index,
                        message=f"Could not read frame {item.frame_index}",
                    )
                )
                continue
            items_by_frame.setdefault(item.frame_index, []).append(item)

        frame_groups = list(items_by_frame.items())
        with self._lock:
            for offset in range(0, len(frame_groups), request.batch_size):
                chunk = frame_groups[offset : offset + request.batch_size]
                chunk_results, chunk_errors = self._segment_frame_chunk(
                    chunk,
                    frames=frames,
                    score_threshold=request.score_threshold,
                )
                results.extend(chunk_results)
                errors.extend(chunk_errors)

        return Sam3SegmentFramesResponse(results=results, errors=errors)

    def _segment_frame_chunk(
        self,
        frame_groups: list[tuple[int, list[Sam3SegmentFrameItem]]],
        *,
        frames: dict[int, Image.Image],
        score_threshold: float,
    ) -> tuple[list[Sam3SegmentFrameResult], list[Sam3SegmentFrameError]]:
        datapoints = []
        query_items: dict[int, Sam3SegmentFrameItem] = {}
        errors: list[Sam3SegmentFrameError] = []

        for frame_index, items in frame_groups:
            datapoint = Datapoint(find_queries=[], images=[])
            self._set_datapoint_image(datapoint, frames[frame_index])
            for item in items:
                try:
                    query_id = self._add_datapoint_seed_query(datapoint, item.seed)
                except ValueError as exc:
                    errors.append(
                        Sam3SegmentFrameError(
                            request_index=item.request_index,
                            frame_index=item.frame_index,
                            message=str(exc),
                        )
                    )
                    continue
                query_items[query_id] = item
            if datapoint.find_queries:
                datapoints.append(self.transform(datapoint))

        if not datapoints:
            return [], errors

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        device = next(self.model.parameters()).device
        batch = copy_data_to_device(batch, device, non_blocking=True)
        with torch.inference_mode():
            output = self.model(batch)
        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=score_threshold,
            to_cpu=False,
        )
        processed_results = postprocessor.process_results(output, batch.find_metadatas)

        results = []
        for query_id, item in query_items.items():
            processed_result = processed_results.get(query_id)
            masks = []
            if processed_result is not None:
                masks = self._processed_result_to_masks(
                    processed_result,
                    max_masks=item.max_masks,
                    source_seed_index=item.request_index,
                )
            results.append(
                Sam3SegmentFrameResult(
                    request_index=item.request_index,
                    frame_index=item.frame_index,
                    masks=masks,
                )
            )

        return results, errors

    def _set_datapoint_image(self, datapoint: Datapoint, image: Image.Image) -> None:
        width, height = image.size
        datapoint.images = [Sam3Image(data=image, objects=[], size=[height, width])]

    def _add_datapoint_seed_query(
        self,
        datapoint: Datapoint,
        seed: Sam3Seed,
    ) -> int:
        if seed.prompt is None:
            raise ValueError("Batched image segmentation only supports text seeds")
        if len(datapoint.images) != 1:
            raise ValueError("Set the image before adding a SAM3 query")

        width, height = datapoint.images[0].size
        query_id = self._query_counter
        self._query_counter += 1
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=seed.prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[width, height],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        return query_id

    def _processed_result_to_masks(
        self,
        result: dict[str, Any],
        *,
        max_masks: int,
        source_seed_index: int,
    ) -> list[Sam3Mask]:
        boxes = result.get("boxes")
        scores = result.get("scores")
        masks = result.get("masks")
        if boxes is None or scores is None:
            return []

        boxes_list = _to_python_list(boxes)
        scores_list = _to_python_list(scores)
        indexes = list(range(len(boxes_list)))
        indexes.sort(key=lambda index: float(scores_list[index]), reverse=True)

        output_masks = []
        for index in indexes[:max_masks]:
            mask_rle = None
            if masks is not None:
                mask_rle = _binary_mask_to_rle(masks[index])
            output_masks.append(
                Sam3Mask(
                    mask_id=str(index),
                    bbox_xyxy=_box_xyxy_tuple(boxes_list[index]),
                    mask_rle=mask_rle,
                    confidence=float(scores_list[index]),
                    source_seed_index=source_seed_index,
                )
            )
        return output_masks

    def _find_video_path(self, video_id: str) -> Path:
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            path = settings.upload_dir / f"{video_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Video {video_id} not found")

    def _open_video_capture(self, video_path: Path) -> cv2.VideoCapture:
        backend = settings.opencv_video_backend.lower()
        if backend != "ffmpeg":
            raise ValueError(f"Unsupported OpenCV video backend: {backend}")
        if settings.opencv_ffmpeg_capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                settings.opencv_ffmpeg_capture_options
            )

        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()
        raise ValueError(f"Could not open video with OpenCV: {video_path}")

    def _read_video_frames(
        self,
        video_path: Path,
        *,
        frame_indexes: set[int],
    ) -> dict[int, Image.Image]:
        frames = {}
        if not frame_indexes:
            return frames

        cap = self._open_video_capture(video_path)
        try:
            for frame_index in sorted(frame_indexes):
                if frame_index < 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[frame_index] = Image.fromarray(rgb_frame)
        finally:
            cap.release()
        return frames


def _binary_mask_to_rle(mask: Any) -> str | None:
    mask_tensor = mask
    if not isinstance(mask_tensor, torch.Tensor):
        mask_tensor = torch.as_tensor(np.asarray(mask_tensor))
    mask_array = mask_tensor.detach().cpu().bool().numpy().squeeze()
    if mask_array.ndim != 2:
        return None
    encoded_mask = mask_utils.encode(np.asfortranarray(mask_array.astype(np.uint8)))
    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
    return json.dumps(
        {
            "encoding": "coco_rle",
            "size": encoded_mask["size"],
            "counts": encoded_mask["counts"],
        },
        separators=(",", ":"),
    )


def _box_xyxy_tuple(box: Any) -> tuple[float, float, float, float]:
    values = _to_python_list(box)
    if len(values) != 4:
        raise ValueError(f"Expected 4 box coordinates, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _to_python_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]
