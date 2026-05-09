from __future__ import annotations

import cv2
import torch
from transformers import Sam3Model, Sam3Processor

from .models import Sam3ExtractRequest, Sam3ExtractResponse, EntityTrack, TrackPoint
from .settings import settings

def _box_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
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

    def process(self, request: Sam3ExtractRequest) -> Sam3ExtractResponse:
        video_exts = [".mp4", ".mov", ".avi", ".mkv"]
        video_path = None
        for ext in video_exts:
            p = settings.upload_dir / f"{request.video_id}{ext}"
            if p.exists():
                video_path = p
                break
        
        if not video_path:
            raise FileNotFoundError(f"Video {request.video_id} not found")

        # Extract frames at 1 fps for tracking
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        frames = []
        timestamps = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Sample at 1 fps
            if frame_idx % int(fps) == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(frame_idx / fps)
            frame_idx += 1
        cap.release()

        tracks = []
        for seed_idx, seed in enumerate(request.seeds):
            track_id = f"seed-{seed_idx}"
            points_out = []
            
            # Simple Tracking: find the closest frame to the seed, start tracking
            start_idx = 0
            min_diff = float("inf")
            for i, ts in enumerate(timestamps):
                if abs(ts - seed.timestamp_seconds) < min_diff:
                    min_diff = abs(ts - seed.timestamp_seconds)
                    start_idx = i
                    
            last_bbox = seed.bbox_xyxy
            
            for i in range(start_idx, len(frames)):
                image = frames[i]
                
                inputs = {}
                if i == start_idx:
                    if seed.prompt:
                        inputs = self.processor(images=image, text=seed.prompt, return_tensors="pt")
                    elif seed.points:
                        input_points = [[[p.x, p.y] for p in seed.points]]
                        input_labels = [[[1 if p.is_positive else 0 for p in seed.points]]]
                        if seed.bbox_xyxy:
                            inputs = self.processor(
                                images=image, 
                                input_points=input_points,
                                input_labels=input_labels,
                                input_boxes=[[[list(seed.bbox_xyxy)]]],
                                return_tensors="pt"
                            )
                        else:
                            inputs = self.processor(
                                images=image, 
                                input_points=input_points,
                                input_labels=input_labels,
                                return_tensors="pt"
                            )
                    elif seed.bbox_xyxy:
                        inputs = self.processor(images=image, input_boxes=[[[list(seed.bbox_xyxy)]]], return_tensors="pt")
                else:
                    if not last_bbox:
                        break
                    # Use last bbox as prompt
                    inputs = self.processor(images=image, input_boxes=[[[list(last_bbox)]]], return_tensors="pt")
                
                if not inputs:
                    break
                    
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                processed = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=request.score_threshold,
                    mask_threshold=request.score_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]
                
                boxes = processed.get("boxes", [])
                scores = processed.get("scores", [])
                
                best_box = None
                best_score = 0.0
                
                for box, score in zip(boxes, scores):
                    box_list = tuple(float(v) for v in box.tolist())
                    if i == start_idx:
                        if score > best_score:
                            best_score = float(score)
                            best_box = box_list
                    else:
                        iou = _box_iou(last_bbox, box_list)
                        combined_score = (iou * 0.7) + (float(score) * 0.3)
                        if combined_score > best_score and combined_score > request.match_threshold:
                            best_score = combined_score
                            best_box = box_list
                            
                if best_box and best_score >= request.high_confidence_threshold:
                    points_out.append(TrackPoint(
                        timestamp_seconds=timestamps[i],
                        bbox_xyxy=best_box,
                        confidence=float(best_score)
                    ))
                    last_bbox = best_box
                else:
                    # Lost track
                    break
                    
            if len(points_out) >= request.min_points:
                tracks.append(EntityTrack(
                    track_id=track_id,
                    points=points_out,
                    confidence=1.0
                ))
                
        return Sam3ExtractResponse(tracks=tracks)
