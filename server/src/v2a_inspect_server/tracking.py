from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Sequence

from v2a_inspect.tools.types import (
    FrameBatch,
    Sam3EntityTrack,
    Sam3TrackPoint,
    Sam3VisualFeatures,
    SampledFrame,
)


@dataclass(frozen=True)
class FrameDetection:
    frame: SampledFrame
    bbox_xyxy: list[float]
    confidence: float
    label_hint: str | None = None
    mask_rle: str | None = None


def dedupe_frame_detections(
    detections: Sequence[FrameDetection], *, iou_threshold: float = 0.8
) -> list[FrameDetection]:
    kept: list[FrameDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(_box_iou(detection.bbox_xyxy, existing.bbox_xyxy) >= iou_threshold for existing in kept):
            continue
        kept.append(detection)
    return kept


def link_frame_detections(
    batch: FrameBatch,
    *,
    detections_by_frame: Sequence[Sequence[FrameDetection]],
    features: Sam3VisualFeatures,
    min_points: int = 2,
    high_confidence_threshold: float = 0.6,
    match_threshold: float = 0.45,
) -> list[Sam3EntityTrack]:
    active_tracks: list[dict[str, object]] = []

    for frame_index, detections in enumerate(detections_by_frame):
        matched_track_indices: set[int] = set()
        for detection in dedupe_frame_detections(detections):
            best_index: int | None = None
            best_score = 0.0
            for track_index, track_state in enumerate(active_tracks):
                if track_index in matched_track_indices:
                    continue
                if int(track_state["last_frame_index"]) < frame_index - 1:
                    continue
                score = _track_match_score(track_state, detection)
                if score > best_score:
                    best_score = score
                    best_index = track_index
            if best_index is not None and best_score >= match_threshold:
                track_state = active_tracks[best_index]
                track_state["points"].append(
                    Sam3TrackPoint(
                        timestamp_seconds=detection.frame.timestamp_seconds,
                        frame_path=detection.frame.image_path,
                        confidence=detection.confidence,
                        bbox_xyxy=list(detection.bbox_xyxy),
                        mask_rle=detection.mask_rle,
                    )
                )
                track_state["last_bbox"] = list(detection.bbox_xyxy)
                track_state["last_frame_index"] = frame_index
                track_state["confidences"].append(detection.confidence)
                track_state["frame_indices"].append(frame_index)
                if detection.label_hint:
                    track_state["labels"].append(detection.label_hint)
                matched_track_indices.add(best_index)
                continue

            active_tracks.append(
                {
                    "track_id": f"scene-{batch.scene_index}-track-{len(active_tracks)}",
                    "points": [
                        Sam3TrackPoint(
                            timestamp_seconds=detection.frame.timestamp_seconds,
                            frame_path=detection.frame.image_path,
                            confidence=detection.confidence,
                            bbox_xyxy=list(detection.bbox_xyxy),
                            mask_rle=detection.mask_rle,
                        )
                    ],
                    "labels": [detection.label_hint] if detection.label_hint else [],
                    "confidences": [detection.confidence],
                    "last_bbox": list(detection.bbox_xyxy),
                    "last_frame_index": frame_index,
                    "frame_indices": [frame_index],
                }
            )
            matched_track_indices.add(len(active_tracks) - 1)

    tracks: list[Sam3EntityTrack] = []
    for track_state in active_tracks:
        points = list(track_state["points"])
        confidences = list(track_state["confidences"])
        if len(points) < min_points and max(confidences, default=0.0) < high_confidence_threshold:
            continue
        labels = Counter(track_state["labels"])
        label_hint = labels.most_common(1)[0][0] if labels else None
        tracks.append(
            Sam3EntityTrack(
                track_id=str(track_state["track_id"]),
                scene_index=batch.scene_index,
                start_seconds=points[0].timestamp_seconds,
                end_seconds=points[-1].timestamp_seconds,
                confidence=round(sum(confidences) / len(confidences), 4),
                label_hint=label_hint,
                points=points,
                features=_track_visual_features(
                    track_state,
                    detections_by_frame=detections_by_frame,
                    scene_features=features,
                ),
            )
        )
    return tracks


def _track_visual_features(
    track_state: dict[str, object],
    *,
    detections_by_frame: Sequence[Sequence[FrameDetection]],
    scene_features: Sam3VisualFeatures,
) -> Sam3VisualFeatures:
    points = list(track_state["points"])
    frame_indices = list(track_state.get("frame_indices", []))
    if not points:
        return scene_features

    trajectory_motion_score = _trajectory_motion_score(points)
    scale_change_score = _scale_change_score(points)
    continuity_score = _continuity_score(frame_indices)
    overlap_score = _overlap_score(points, frame_indices, detections_by_frame)
    local_crowd_score = _local_crowd_score(frame_indices, detections_by_frame)
    motion_score = min(1.0, (trajectory_motion_score * 0.7) + (scale_change_score * 0.3))
    interaction_score = min(
        1.0,
        (overlap_score * 0.55) + (local_crowd_score * 0.25) + (scale_change_score * 0.2),
    )
    crowd_score = min(1.0, max(local_crowd_score, scene_features.crowd_score * 0.35))

    return Sam3VisualFeatures(
        motion_score=round(motion_score, 4),
        interaction_score=round(interaction_score, 4),
        crowd_score=round(crowd_score, 4),
        camera_dynamics_score=scene_features.camera_dynamics_score,
        trajectory_motion_score=round(trajectory_motion_score, 4),
        scale_change_score=round(scale_change_score, 4),
        continuity_score=round(continuity_score, 4),
        overlap_score=round(overlap_score, 4),
        scene_crowd_score=scene_features.crowd_score,
        scene_camera_dynamics_score=scene_features.camera_dynamics_score,
    )


def _track_match_score(track_state: dict[str, object], detection: FrameDetection) -> float:
    iou = _box_iou(track_state["last_bbox"], detection.bbox_xyxy)
    labels: list[str] = [label for label in track_state["labels"] if label]
    label_bonus = 0.0
    if detection.label_hint and detection.label_hint in labels:
        label_bonus = 0.25
    elif not labels or detection.label_hint is None:
        label_bonus = 0.1
    return min(1.0, (iou * 0.75) + label_bonus)


def _trajectory_motion_score(points: Sequence[Sam3TrackPoint]) -> float:
    if len(points) < 2:
        return 0.0
    displacements: list[float] = []
    for left, right in zip(points, points[1:], strict=False):
        if left.bbox_xyxy is None or right.bbox_xyxy is None:
            continue
        left_cx, left_cy = _box_center(left.bbox_xyxy)
        right_cx, right_cy = _box_center(right.bbox_xyxy)
        norm = max(_box_diagonal(left.bbox_xyxy), _box_diagonal(right.bbox_xyxy), 1.0)
        displacements.append((((right_cx - left_cx) ** 2 + (right_cy - left_cy) ** 2) ** 0.5) / norm)
    if not displacements:
        return 0.0
    return min(sum(displacements) / len(displacements), 1.0)


def _scale_change_score(points: Sequence[Sam3TrackPoint]) -> float:
    if len(points) < 2:
        return 0.0
    deltas: list[float] = []
    for left, right in zip(points, points[1:], strict=False):
        if left.bbox_xyxy is None or right.bbox_xyxy is None:
            continue
        left_area = _box_area(left.bbox_xyxy)
        right_area = _box_area(right.bbox_xyxy)
        denom = max(left_area, right_area, 1.0)
        deltas.append(abs(right_area - left_area) / denom)
    if not deltas:
        return 0.0
    return min(sum(deltas) / len(deltas), 1.0)


def _continuity_score(frame_indices: Sequence[int]) -> float:
    if len(frame_indices) < 2:
        return 0.0
    gaps = [max(right - left, 1) for left, right in zip(frame_indices, frame_indices[1:], strict=False)]
    average_gap = sum(gaps) / len(gaps)
    return min(1.0 / average_gap, 1.0)


def _overlap_score(
    points: Sequence[Sam3TrackPoint],
    frame_indices: Sequence[int],
    detections_by_frame: Sequence[Sequence[FrameDetection]],
) -> float:
    overlaps: list[float] = []
    for point, frame_index in zip(points, frame_indices, strict=False):
        if point.bbox_xyxy is None or frame_index >= len(detections_by_frame):
            continue
        other_ious = [
            _box_iou(point.bbox_xyxy, detection.bbox_xyxy)
            for detection in detections_by_frame[frame_index]
            if detection.bbox_xyxy != point.bbox_xyxy
        ]
        overlaps.append(max(other_ious, default=0.0))
    if not overlaps:
        return 0.0
    return min(sum(overlaps) / len(overlaps), 1.0)


def _local_crowd_score(
    frame_indices: Sequence[int],
    detections_by_frame: Sequence[Sequence[FrameDetection]],
) -> float:
    if not frame_indices:
        return 0.0
    counts = []
    for frame_index in frame_indices:
        if frame_index >= len(detections_by_frame):
            continue
        counts.append(max(len(detections_by_frame[frame_index]) - 1, 0) / 3.0)
    if not counts:
        return 0.0
    return min(sum(counts) / len(counts), 1.0)


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


def _box_center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _box_diagonal(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return ((max(x2 - x1, 0.0) ** 2) + (max(y2 - y1, 0.0) ** 2)) ** 0.5
