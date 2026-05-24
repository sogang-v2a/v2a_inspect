from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from v2a_inspect.media_utils.video import extract_frame
from v2a_inspect.models import SoundTimeline
from v2a_inspect.visualization.colors import color_for_index
from v2a_inspect.visualization.drawing import draw_bbox, draw_frame_index, draw_label

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor


class SoundTimelineReadTools:
    def __init__(self, editor: SoundTimelineEditor) -> None:
        self.editor = editor

    def list_scenes(self) -> dict[str, Any]:
        video_asset = self.editor.video_asset
        return {
            "frame_count": video_asset.frame_count,
            "fps": video_asset.fps,
            "scenes": [
                {
                    "scene_index": scene_index,
                    "initial_scene_id": str(scene.initial_scene_id),
                    "start_frame_index": scene.start_frame_index,
                    "end_frame_index": scene.end_frame_index,
                    "frame_count": scene.frame_count,
                    "keyframe_indexes": [
                        keyframe.frame_index for keyframe in scene.keyframes
                    ],
                    "track_count": len(scene.scene_tracks),
                    "object_seed_count": 0
                    if scene.initial_analysis is None
                    else len(scene.initial_analysis.object_seeds),
                }
                for scene_index, scene in enumerate(video_asset.initial_scenes)
            ],
        }

    def get_scene_summary(self, scene_index: int) -> dict[str, Any]:
        scene = self._scene(scene_index)
        object_seeds: list[dict[str, Any]] = []
        if scene.initial_analysis is not None:
            object_seeds = [
                {
                    "label": seed.label,
                    "tracking_prompt": seed.tracking_prompt,
                    "notes": seed.notes,
                }
                for seed in scene.initial_analysis.object_seeds
            ]
        return {
            "scene_index": scene_index,
            "initial_scene_id": str(scene.initial_scene_id),
            "start_frame_index": scene.start_frame_index,
            "end_frame_index": scene.end_frame_index,
            "frame_count": scene.frame_count,
            "keyframe_indexes": [keyframe.frame_index for keyframe in scene.keyframes],
            "object_seeds": object_seeds,
            "tracks": self.list_tracks(scene_index)["tracks"],
        }

    def get_annotated_frame(self, frame_index: int) -> dict[str, Any]:
        self.editor.check_frame_index(frame_index, allow_end=False)
        image = extract_frame(Path(self.editor.video_asset.source_path), frame_index)
        tracks = self._tracks_at_frame(frame_index)
        for track_index, track_view in enumerate(tracks):
            color = color_for_index(track_index)
            bbox = track_view["bbox_xyxy"]
            if bbox is not None:
                draw_bbox(image, tuple(bbox), color)
                label_position = (int(bbox[0]), max(0, int(bbox[1]) - 16))
            else:
                label_position = (10, 28 + (track_index * 18))
            draw_label(
                image,
                label_position,
                f"{track_view['scene_track_id'][:8]} {track_view['confidence']:.2f}",
                color,
            )
        draw_frame_index(image, frame_index)
        return {
            "frame_index": frame_index,
            "image": _image_data_url(image),
            "tracks": tracks,
        }

    def list_tracks(self, scene_index: int) -> dict[str, Any]:
        scene = self._scene(scene_index)
        return {
            "scene_index": scene_index,
            "tracks": [
                {
                    "scene_track_id": str(track.scene_track_id),
                    "tracking_prompt": track.tracking_prompt,
                    "source_label": None
                    if track.source_object_seed is None
                    else track.source_object_seed.label,
                    "start_frame_index": track.start_frame_index,
                    "end_frame_index": track.end_frame_index,
                    "frame_count": track.frame_count,
                    "confidence": track.confidence,
                    "notes": track.notes,
                }
                for track in scene.scene_tracks
            ],
        }

    def get_visual_events(
        self,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
    ) -> dict[str, Any]:
        layer = self.editor.video_asset.visual_identity_layer
        if layer is None:
            return {"visual_events": []}
        events = []
        for event in layer.visual_events:
            if (
                start_frame_index is not None
                and event.end_frame_index <= start_frame_index
            ):
                continue
            if (
                end_frame_index is not None
                and event.start_frame_index >= end_frame_index
            ):
                continue
            events.append(event.model_dump(mode="json"))
        return {"visual_events": events}

    def get_sound_timeline(self) -> dict[str, Any]:
        timeline = self.editor.video_asset.sound_timeline
        if timeline is None:
            return SoundTimeline().model_dump(mode="json")
        return timeline.model_dump(mode="json")

    def _scene(self, scene_index: int):
        scenes = self.editor.video_asset.initial_scenes
        if scene_index < 0 or scene_index >= len(scenes):
            raise ValueError(f"scene_index out of range: {scene_index}")
        return scenes[scene_index]

    def _tracks_at_frame(self, frame_index: int) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        for scene_index, scene in enumerate(self.editor.video_asset.initial_scenes):
            if not (scene.start_frame_index <= frame_index < scene.end_frame_index):
                continue
            for track in scene.scene_tracks:
                for point in track.points:
                    if point.frame_index != frame_index:
                        continue
                    tracks.append(
                        {
                            "scene_index": scene_index,
                            "scene_track_id": str(track.scene_track_id),
                            "tracking_prompt": track.tracking_prompt,
                            "source_label": None
                            if track.source_object_seed is None
                            else track.source_object_seed.label,
                            "bbox_xyxy": point.bbox_xyxy,
                            "confidence": point.confidence,
                        }
                    )
        return tracks


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
