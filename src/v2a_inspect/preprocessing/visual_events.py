from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Literal
from uuid import UUID

import numpy as np

from v2a_inspect.media_utils import decode_coco_rle
from v2a_inspect.models import (
    SceneTrack,
    SceneTrackPoint,
    VideoAsset,
    VisualEvent,
    VisualObject,
)

MIN_RELIABLE_POINTS = 3
LOW_MEAN_CONFIDENCE = 0.35
STATIONARY_DISPLACEMENT_RATIO = 0.03
FAST_DISPLACEMENT_RATIO = 0.25
SCALE_CHANGE_RATIO = 1.5
CONTACT_OVERLAP_RATIO = 0.2
CONTACT_MIN_CONFIDENCE = 0.25
CONTACT_MIN_SHARED_FRAMES = 5
MOTION_SMOOTHING_WINDOW = 5


@dataclass(frozen=True)
class _MaskSample:
    frame_index: int
    mask: np.ndarray
    mask_area: float
    centroid_x: float
    centroid_y: float
    confidence: float


@dataclass(frozen=True)
class _VisualObjectTrack:
    visual_object: VisualObject
    scene_track: SceneTrack
    samples: list[_MaskSample]


def compute_visual_events(video_asset: VideoAsset) -> VideoAsset:
    """Recompute visual events from current visual objects and mask time series."""

    if video_asset.visual_identity_layer is None:
        raise ValueError("video_asset must have a visual_identity_layer")

    scene_tracks = _scene_tracks_by_id(video_asset)
    visual_object_tracks = _visual_object_tracks(
        video_asset.visual_identity_layer.visual_objects,
        scene_tracks,
    )

    visual_events: list[VisualEvent] = []
    for visual_object_track in visual_object_tracks:
        visual_events.extend(_object_visual_events(visual_object_track))
    visual_events.extend(_contact_visual_events(visual_object_tracks))

    visual_identity_layer = video_asset.visual_identity_layer.model_copy(
        update={"visual_events": visual_events}
    )
    return video_asset.model_copy(
        update={"visual_identity_layer": visual_identity_layer}
    )


def _scene_tracks_by_id(video_asset: VideoAsset) -> dict[UUID, SceneTrack]:
    scene_tracks: dict[UUID, SceneTrack] = {}
    for initial_scene in video_asset.initial_scenes:
        for scene_track in initial_scene.scene_tracks:
            scene_tracks[scene_track.scene_track_id] = scene_track
    return scene_tracks


def _visual_object_tracks(
    visual_objects: list[VisualObject],
    scene_tracks: dict[UUID, SceneTrack],
) -> list[_VisualObjectTrack]:
    visual_object_tracks: list[_VisualObjectTrack] = []
    for visual_object in visual_objects:
        for presence in visual_object.presences:
            if presence.scene_track_id is None:
                continue
            scene_track = scene_tracks.get(presence.scene_track_id)
            if scene_track is None:
                continue
            visual_object_tracks.append(
                _VisualObjectTrack(
                    visual_object=visual_object,
                    scene_track=scene_track,
                    samples=_track_samples(scene_track),
                )
            )
    return visual_object_tracks


def _track_samples(scene_track: SceneTrack) -> list[_MaskSample]:
    samples: list[_MaskSample] = []
    points = sorted(scene_track.points, key=lambda point: point.frame_index)
    for point in points:
        sample = _mask_sample_from_point(point)
        if sample is not None:
            samples.append(sample)
    return samples


def _mask_sample_from_point(point: SceneTrackPoint) -> _MaskSample | None:
    mask = _decode_point_mask(point)
    if mask is None:
        return None

    mask_area = _mask_area(mask)
    if mask_area <= 0:
        return None

    centroid_x, centroid_y = _mask_centroid(mask)
    return _MaskSample(
        frame_index=point.frame_index,
        mask=mask,
        mask_area=mask_area,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        confidence=point.confidence,
    )


def _object_visual_events(
    visual_object_track: _VisualObjectTrack,
) -> list[VisualEvent]:
    if not visual_object_track.samples:
        return [_no_usable_masks_event(visual_object_track)]

    visual_object_id = visual_object_track.visual_object.visual_object_id
    samples = visual_object_track.samples
    start_frame_index = samples[0].frame_index
    end_frame_index = samples[-1].frame_index + 1
    mean_confidence = _mean_confidence(samples)

    events = [
        VisualEvent(
            visual_object_id=visual_object_id,
            start_frame_index=start_frame_index,
            end_frame_index=start_frame_index + 1,
            event_type="appearance",
            description="appears",
            confidence=mean_confidence,
        ),
        VisualEvent(
            visual_object_id=visual_object_id,
            start_frame_index=samples[-1].frame_index,
            end_frame_index=end_frame_index,
            event_type="disappearance",
            description="disappears",
            confidence=mean_confidence,
        ),
        VisualEvent(
            visual_object_id=visual_object_id,
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            event_type=_motion_event_type(samples),
            description=_motion_description(samples),
            confidence=mean_confidence,
        ),
    ]

    scale_change_description = _scale_change_description(samples)
    if scale_change_description is not None:
        events.append(
            VisualEvent(
                visual_object_id=visual_object_id,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                event_type="scale_change",
                description=scale_change_description,
                confidence=mean_confidence,
            )
        )

    if len(samples) < MIN_RELIABLE_POINTS or mean_confidence < LOW_MEAN_CONFIDENCE:
        events.append(
            VisualEvent(
                visual_object_id=visual_object_id,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                event_type="uncertain",
                description="sparse or low-confidence masks",
                confidence=1.0,
            )
        )

    return events


def _no_usable_masks_event(visual_object_track: _VisualObjectTrack) -> VisualEvent:
    return VisualEvent(
        visual_object_id=visual_object_track.visual_object.visual_object_id,
        start_frame_index=visual_object_track.scene_track.start_frame_index,
        end_frame_index=visual_object_track.scene_track.end_frame_index,
        event_type="uncertain",
        description="no usable masks",
        confidence=1.0,
    )


def _contact_visual_events(
    visual_object_tracks: list[_VisualObjectTrack],
) -> list[VisualEvent]:
    events: list[VisualEvent] = []
    usable_tracks = [track for track in visual_object_tracks if track.samples]
    for left_index, left in enumerate(usable_tracks):
        for right in usable_tracks[left_index + 1 :]:
            contact = _contact_interval(left.samples, right.samples)
            if contact is None:
                continue
            start_frame_index, end_frame_index, confidence = contact
            events.append(
                VisualEvent(
                    visual_object_id=left.visual_object.visual_object_id,
                    related_visual_object_ids=[right.visual_object.visual_object_id],
                    start_frame_index=start_frame_index,
                    end_frame_index=end_frame_index,
                    event_type="contact",
                    description="overlaps another object",
                    confidence=confidence,
                )
            )
            events.append(
                VisualEvent(
                    visual_object_id=right.visual_object.visual_object_id,
                    related_visual_object_ids=[left.visual_object.visual_object_id],
                    start_frame_index=start_frame_index,
                    end_frame_index=end_frame_index,
                    event_type="contact",
                    description="overlaps another object",
                    confidence=confidence,
                )
            )
    return events


def _contact_interval(
    left_samples: list[_MaskSample],
    right_samples: list[_MaskSample],
) -> tuple[int, int, float] | None:
    right_by_frame = {sample.frame_index: sample for sample in right_samples}
    contact_runs: list[list[tuple[int, float]]] = []
    current_run: list[tuple[int, float]] = []
    previous_contact_frame: int | None = None
    for left_sample in left_samples:
        right_sample = right_by_frame.get(left_sample.frame_index)
        if right_sample is None:
            continue
        overlap = _mask_overlap_over_smaller(left_sample.mask, right_sample.mask)
        if overlap < CONTACT_OVERLAP_RATIO:
            if current_run:
                contact_runs.append(current_run)
                current_run = []
            previous_contact_frame = None
            continue
        if (
            previous_contact_frame is not None
            and left_sample.frame_index != previous_contact_frame + 1
            and current_run
        ):
            contact_runs.append(current_run)
            current_run = []
        current_run.append((left_sample.frame_index, overlap))
        previous_contact_frame = left_sample.frame_index

    if current_run:
        contact_runs.append(current_run)

    qualifying_runs = [
        run for run in contact_runs if len(run) >= CONTACT_MIN_SHARED_FRAMES
    ]
    if not qualifying_runs:
        return None

    best_run = max(
        qualifying_runs,
        key=lambda run: _mean([overlap for _, overlap in run]),
    )
    confidence = min(_mean([overlap for _, overlap in best_run]), 1.0)
    if confidence < CONTACT_MIN_CONFIDENCE:
        return None
    return best_run[0][0], best_run[-1][0] + 1, confidence


def _mask_overlap_over_smaller(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    intersection = float(np.logical_and(left_mask, right_mask).sum())
    smaller_area = min(_mask_area(left_mask), _mask_area(right_mask))
    if smaller_area <= 0:
        return 0.0
    return intersection / smaller_area


def _motion_event_type(
    samples: list[_MaskSample],
) -> Literal["stationary", "motion", "fast_motion"]:
    motion_ratio = _motion_ratio(samples)
    if motion_ratio <= STATIONARY_DISPLACEMENT_RATIO:
        return "stationary"
    if motion_ratio >= FAST_DISPLACEMENT_RATIO:
        return "fast_motion"
    return "motion"


def _motion_description(samples: list[_MaskSample]) -> str:
    event_type = _motion_event_type(samples)
    if event_type == "stationary":
        return "stationary"

    direction = _motion_direction(samples)
    if event_type == "fast_motion":
        return f"moves quickly {direction}"
    return f"moves {direction}"


def _motion_direction(samples: list[_MaskSample]) -> str:
    first_x, first_y = _edge_centroid(samples, from_start=True)
    last_x, last_y = _edge_centroid(samples, from_start=False)
    dx = last_x - first_x
    dy = last_y - first_y
    if abs(dx) >= abs(dy) * 1.5:
        if dx >= 0:
            return "right"
        return "left"
    if abs(dy) >= abs(dx) * 1.5:
        if dy >= 0:
            return "down"
        return "up"
    if dx >= 0 and dy >= 0:
        return "down-right"
    if dx >= 0:
        return "up-right"
    if dy >= 0:
        return "down-left"
    return "up-left"


def _displacement_ratio(samples: list[_MaskSample]) -> float:
    return _motion_ratio(samples)


def _motion_ratio(samples: list[_MaskSample]) -> float:
    if len(samples) < 2:
        return 0.0
    smoothed_centroids = _smoothed_centroids(samples)
    path_displacement = 0.0
    for left_centroid, right_centroid in zip(
        smoothed_centroids,
        smoothed_centroids[1:],
        strict=False,
    ):
        path_displacement += hypot(
            right_centroid[0] - left_centroid[0],
            right_centroid[1] - left_centroid[1],
        )
    first_x, first_y = _edge_centroid(samples, from_start=True)
    last_x, last_y = _edge_centroid(samples, from_start=False)
    net_displacement = hypot(last_x - first_x, last_y - first_y)
    diagonal = _mask_diagonal(samples)
    if diagonal <= 0:
        return 0.0
    return max(net_displacement, path_displacement * 0.5) / diagonal


def _smoothed_centroids(samples: list[_MaskSample]) -> list[tuple[float, float]]:
    smoothed: list[tuple[float, float]] = []
    radius = MOTION_SMOOTHING_WINDOW // 2
    for index in range(len(samples)):
        start_index = max(0, index - radius)
        end_index = min(len(samples), index + radius + 1)
        window = samples[start_index:end_index]
        smoothed.append(
            (
                _mean([sample.centroid_x for sample in window]),
                _mean([sample.centroid_y for sample in window]),
            )
        )
    return smoothed


def _edge_centroid(
    samples: list[_MaskSample],
    *,
    from_start: bool,
) -> tuple[float, float]:
    count = max(1, min(len(samples), MOTION_SMOOTHING_WINDOW))
    if from_start:
        edge_samples = samples[:count]
    else:
        edge_samples = samples[-count:]
    return (
        _mean([sample.centroid_x for sample in edge_samples]),
        _mean([sample.centroid_y for sample in edge_samples]),
    )


def _mask_diagonal(samples: list[_MaskSample]) -> float:
    height, width = samples[0].mask.shape[:2]
    return hypot(float(width), float(height))


def _scale_change_description(samples: list[_MaskSample]) -> str | None:
    areas = [sample.mask_area for sample in samples if sample.mask_area > 0]
    if len(areas) < 2:
        return None

    first_area = _edge_mean(areas, from_start=True)
    last_area = _edge_mean(areas, from_start=False)
    if first_area <= 0 or last_area <= 0:
        return None

    ratio = last_area / first_area
    if ratio >= SCALE_CHANGE_RATIO:
        return "grows larger"
    if ratio <= 1 / SCALE_CHANGE_RATIO:
        return "shrinks"
    return None


def _edge_mean(values: list[float], *, from_start: bool) -> float:
    count = max(1, len(values) // 3)
    if from_start:
        return _mean(values[:count])
    return _mean(values[-count:])


def _mask_area(mask: np.ndarray) -> float:
    return float(mask.sum())


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    return float(cols.mean()), float(rows.mean())


def _decode_point_mask(point: SceneTrackPoint) -> np.ndarray | None:
    if point.mask is None or point.mask.rle is None:
        return None
    return decode_coco_rle(point.mask.rle)


def _mean_confidence(samples: list[_MaskSample]) -> float:
    return _mean([sample.confidence for sample in samples])


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
