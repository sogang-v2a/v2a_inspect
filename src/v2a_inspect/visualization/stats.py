from __future__ import annotations

from typing import Any

from v2a_inspect.client.models import Sam3TrackVideoResponse
from v2a_inspect.models import VideoAsset


def summarize_tracks(tracks: Any) -> list[dict[str, int | float | str]]:
    track_list = _normalize_tracks(tracks)
    rows = []
    for track in track_list:
        points = list(track.points)
        confidences = [float(point.confidence) for point in points]
        rows.append(
            {
                "track_id": str(
                    getattr(track, "track_id", getattr(track, "scene_track_id", ""))
                ),
                "point_count": len(points),
                "start_frame_index": min(point.frame_index for point in points),
                "end_frame_index": max(point.frame_index for point in points) + 1,
                "min_confidence": min(confidences),
                "mean_confidence": sum(confidences) / len(confidences),
                "mask_count": sum(1 for point in points if _has_mask(point)),
            }
        )
    return rows


def summarize_scenes(video_asset: VideoAsset) -> list[dict[str, int | float]]:
    rows = []
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        rows.append(
            {
                "scene_index": scene_index,
                "start_frame_index": scene.start_frame_index,
                "end_frame_index": scene.end_frame_index,
                "frame_count": scene.frame_count,
                "duration_sec": scene.frame_count / video_asset.fps,
                "keyframe_count": len(scene.keyframes),
                "track_count": len(scene.scene_tracks),
            }
        )
    return rows


def _normalize_tracks(tracks: Any) -> list[Any]:
    if isinstance(tracks, Sam3TrackVideoResponse):
        return list(tracks.tracks)
    return list(tracks)


def _has_mask(point: Any) -> bool:
    return (
        getattr(point, "mask_rle", None) is not None
        or getattr(point, "mask", None) is not None
    )
