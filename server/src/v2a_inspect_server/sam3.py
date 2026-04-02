from __future__ import annotations

from v2a_inspect.tools.types import (
    FrameBatch,
    Sam3EntityTrack,
    Sam3TrackPoint,
    Sam3TrackSet,
    Sam3VisualFeatures,
)

from .image_features import frame_motion_score, summarize_image_paths


class Sam3Client:
    def extract_entities(self, frame_batches: list[FrameBatch]) -> Sam3TrackSet:
        tracks: list[Sam3EntityTrack] = []
        for batch in frame_batches:
            image_paths = [frame.image_path for frame in batch.frames]
            if not image_paths:
                continue
            stats = summarize_image_paths(image_paths)
            motion_score = frame_motion_score(image_paths)
            confidence = _clamp01(
                0.45
                + (motion_score * 0.25)
                + (stats.edge_density * 0.20)
                + (stats.contrast * 0.10)
            )
            features = Sam3VisualFeatures(
                motion_score=motion_score,
                interaction_score=_clamp01(
                    (stats.edge_density * 0.55) + (motion_score * 0.45)
                ),
                crowd_score=_clamp01(
                    (stats.edge_density * 0.60) + (stats.colorfulness * 0.40)
                ),
                camera_dynamics_score=_clamp01(
                    (motion_score * 0.60) + (stats.horizontal_energy * 0.40)
                ),
            )
            start_seconds = batch.frames[0].timestamp_seconds
            end_seconds = batch.frames[-1].timestamp_seconds
            label_hint = _label_hint(stats=stats, motion_score=motion_score)
            points = [
                Sam3TrackPoint(timestamp_seconds=frame.timestamp_seconds)
                for frame in batch.frames
            ]
            tracks.append(
                Sam3EntityTrack(
                    track_id=f"scene-{batch.scene_index}-track-0",
                    scene_index=batch.scene_index,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    confidence=confidence,
                    label_hint=label_hint,
                    points=points,
                    features=features,
                )
            )
        return Sam3TrackSet(provider="native-sam3", strategy="prompt_free", tracks=tracks)

    def recover_with_text_prompt(
        self,
        frame_batches: list[FrameBatch],
        *,
        text_prompt: str,
    ) -> Sam3TrackSet:
        track_set = self.extract_entities(frame_batches)
        if text_prompt.strip():
            for track in track_set.tracks:
                if track.label_hint is None:
                    track.label_hint = text_prompt.strip().split()[0].lower()
        track_set.strategy = "text_recovery"
        return track_set


Sam3RunpodClient = Sam3Client


def _label_hint(*, stats: object, motion_score: float) -> str:
    edge_density = getattr(stats, "edge_density", 0.0)
    colorfulness = getattr(stats, "colorfulness", 0.0)
    horizontal_energy = getattr(stats, "horizontal_energy", 0.0)
    vertical_energy = getattr(stats, "vertical_energy", 0.0)
    if edge_density < 0.12 and motion_score < 0.08:
        return "background"
    if horizontal_energy > vertical_energy and motion_score > 0.18:
        return "vehicle"
    if vertical_energy >= horizontal_energy and colorfulness > 0.15:
        return "person"
    return "object"


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
