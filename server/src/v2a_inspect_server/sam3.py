from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor

from v2a_inspect.tools.types import (
    FrameBatch,
    Sam3EntityTrack,
    Sam3TrackPoint,
    Sam3TrackSet,
    Sam3VisualFeatures,
)

from .image_features import frame_motion_score, summarize_image_paths
from .model_runtime import inference_device, inference_dtype, load_rgb_images, move_inputs_to_device

DEFAULT_SAM3_PROMPTS = [
    "person",
    "vehicle",
    "animal",
    "object",
]


class Sam3Client:
    def __init__(self, *, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.device = inference_device()
        self.dtype = inference_dtype()
        self.processor = Sam3Processor.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = Sam3Model.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def extract_entities(
        self,
        frame_batches: list[FrameBatch],
        *,
        prompts_by_scene: Mapping[int, Sequence[str]] | None = None,
    ) -> Sam3TrackSet:
        tracks: list[Sam3EntityTrack] = []
        for batch in frame_batches:
            batch_tracks = self._extract_scene_tracks(
                batch,
                prompts=prompts_by_scene.get(batch.scene_index) if prompts_by_scene else None,
            )
            tracks.extend(batch_tracks)
        return Sam3TrackSet(provider="sam3", strategy="prompt_free", tracks=tracks)

    def recover_with_text_prompt(
        self,
        frame_batches: list[FrameBatch],
        *,
        text_prompt: str,
    ) -> Sam3TrackSet:
        prompt = text_prompt.strip() or "object"
        prompts_by_scene = {
            batch.scene_index: [prompt]
            for batch in frame_batches
        }
        track_set = self.extract_entities(
            frame_batches,
            prompts_by_scene=prompts_by_scene,
        )
        track_set.strategy = "text_recovery"
        return track_set

    def _extract_scene_tracks(
        self,
        batch: FrameBatch,
        *,
        prompts: Sequence[str] | None,
    ) -> list[Sam3EntityTrack]:
        if not batch.frames:
            return []
        prompt_list = [prompt.strip().lower() for prompt in (prompts or DEFAULT_SAM3_PROMPTS) if prompt.strip()]
        if not prompt_list:
            prompt_list = list(DEFAULT_SAM3_PROMPTS)
        representative_frame = batch.frames[0]
        images = load_rgb_images([representative_frame.image_path])
        image = images[0]
        tracks: list[Sam3EntityTrack] = []
        seen_boxes: list[list[float]] = []
        stats = summarize_image_paths([frame.image_path for frame in batch.frames])
        motion_score = frame_motion_score([frame.image_path for frame in batch.frames])

        for prompt in prompt_list:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            model_inputs = move_inputs_to_device(dict(inputs), self.device)
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            processed = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            boxes = processed.get("boxes", [])
            scores = processed.get("scores", [])
            for index, box in enumerate(boxes):
                score = float(scores[index]) if index < len(scores) else 0.0
                if score < 0.35:
                    continue
                box_list = [float(value) for value in box.tolist()]
                if _is_duplicate_box(box_list, seen_boxes):
                    continue
                seen_boxes.append(box_list)
                tracks.append(
                    Sam3EntityTrack(
                        track_id=f"scene-{batch.scene_index}-track-{len(tracks)}",
                        scene_index=batch.scene_index,
                        start_seconds=batch.frames[0].timestamp_seconds,
                        end_seconds=batch.frames[-1].timestamp_seconds,
                        confidence=min(max(score, 0.0), 1.0),
                        label_hint=prompt,
                        points=[
                            Sam3TrackPoint(
                                timestamp_seconds=frame.timestamp_seconds,
                                bbox_xyxy=box_list,
                            )
                            for frame in batch.frames
                        ],
                        features=Sam3VisualFeatures(
                            motion_score=motion_score,
                            interaction_score=min(max(stats.edge_density, 0.0), 1.0),
                            crowd_score=min(max(stats.colorfulness, 0.0), 1.0),
                            camera_dynamics_score=min(max(stats.horizontal_energy, 0.0), 1.0),
                        ),
                    )
                )
        return tracks


def _is_duplicate_box(candidate: list[float], existing_boxes: Sequence[list[float]]) -> bool:
    return any(_box_iou(candidate, existing) >= 0.8 for existing in existing_boxes)


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom
