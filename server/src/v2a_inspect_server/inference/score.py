from __future__ import annotations

import cv2
import torch
from transformers import AutoProcessor, AutoModel

from ..models import LabelScore, LabelScoreRequest, LabelScoreResponse
from ..settings import settings

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

class Siglip2InferenceClient:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(settings.label_model_id)
        self.model = AutoModel.from_pretrained(
            settings.label_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def score(self, request: LabelScoreRequest) -> LabelScoreResponse:
        # We need to extract frames for each point in the track and run SigLIP on each frame
        # Then average the scores per label? Or take max? The original implementation
        # in tool_registry.py scored each image independently and then returned a list
        # of LabelCandidate per track. It did not average; it returned the scores for
        # each image. However, the LabelScoreResponse expects a list of LabelScore per
        # track_id (not per image). We need to decide how to aggregate.
        # Looking at the original tool_registry.py, the score_track_labels function
        # returns a dict[track_id, list[LabelCandidate]]. Each LabelCandidate corresponds
        # to a label, and the score is the score from the model for that label on the
        # image? Actually, looking at the code:
        #   for score in tooling_runtime.label_client.score_image_labels(
        #       image_paths=image_paths, labels=label_set
        #   )
        # This returns a list of LabelScore (one per label) for the set of images.
        # The label_client.score_image_labels takes a list of image_paths and a list of
        # labels, and returns a list of LabelScore (one per label) where the score is
        # averaged over the images? Let's check the original implementation in
        # server/src/old/embeddings.py for the LabelClient.score_image_labels method.
        # We have that code from earlier: it computes the mean over images for each label.
        # So the original implementation averaged the scores over the images for each label.
        # We'll do the same: for each label, compute the average score over all frames
        # (or over all points? The request has points, which are timestamps with bboxes).
        # We'll extract a frame for each point, run SigLIP on that frame, and then average
        # the scores for each label across all frames.
        
        # We'll accumulate scores per label
        label_scores_sum = {label: 0.0 for label in request.labels}
        label_counts = {label: 0 for label in request.labels}
        
        for point in request.points:
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
                continue
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare text inputs: "This is a photo of {label}."
            texts = [f"This is a photo of {label}." for label in request.labels]
            
            # Process with SigLIP
            inputs = self.processor(
                text=texts,
                images=rgb_frame,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # SigLIP2 model returns logits_per_image (shape: [1, num_texts])
            logits_per_image = outputs.logits_per_image  # we have one image
            probs = torch.sigmoid(logits_per_image).cpu().numpy().flatten()
            
            for idx, label in enumerate(request.labels):
                label_scores_sum[label] += float(probs[idx])
                label_counts[label] += 1
        
        # Compute average scores
        scores = []
        for label in request.labels:
            if label_counts[label] > 0:
                avg_score = label_scores_sum[label] / label_counts[label]
            else:
                avg_score = 0.0
            scores.append(
                LabelScore(
                    label=label,
                    score=avg_score,
                )
            )
        
        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        
        return LabelScoreResponse(track_id=request.track_id, scores=scores)