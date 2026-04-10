from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor

from v2a_inspect.tools.types import (
    FrameBatch,
    Sam3EntityTrack,
    Sam3TrackSet,
    Sam3VisualFeatures,
)

from .image_features import frame_motion_score, summarize_image_paths
from .model_runtime import inference_device, inference_dtype, load_rgb_images, move_inputs_to_device
from .tracking import FrameDetection, link_frame_detections

DEFAULT_SAM3_PROMPTS = [
    "person",
    "cat",
    "dog",
    "car",
    "boat",
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
        strategy = "prompt_seeded" if prompts_by_scene else "prompt_free"
        for batch in frame_batches:
            batch_tracks = self._extract_scene_tracks(
                batch,
                prompts=prompts_by_scene.get(batch.scene_index) if prompts_by_scene else None,
            )
            tracks.extend(batch_tracks)
        return Sam3TrackSet(provider="sam3", strategy=strategy, tracks=tracks)

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
        detections_by_frame: list[list[FrameDetection]] = []
        stats = summarize_image_paths([frame.image_path for frame in batch.frames])
        motion_score = frame_motion_score([frame.image_path for frame in batch.frames])
        features = Sam3VisualFeatures(
            motion_score=motion_score,
            interaction_score=min(max(stats.edge_density, 0.0), 1.0),
            crowd_score=min(max(stats.colorfulness, 0.0), 1.0),
            camera_dynamics_score=min(max(stats.horizontal_energy, 0.0), 1.0),
        )

        for frame in batch.frames:
            images = load_rgb_images([frame.image_path])
            image = images[0]
            frame_detections: list[FrameDetection] = []
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
                    frame_detections.append(
                        FrameDetection(
                            frame=frame,
                            bbox_xyxy=[float(value) for value in box.tolist()],
                            confidence=min(max(score, 0.0), 1.0),
                            label_hint=prompt,
                        )
                    )
            detections_by_frame.append(frame_detections)
        return link_frame_detections(
            batch,
            detections_by_frame=detections_by_frame,
            features=features,
        )
