from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from v2a_inspect.media_utils.video import extract_frame
from v2a_inspect.visualization.colors import color_for_index
from v2a_inspect.visualization.drawing import draw_bbox, draw_frame_index, draw_label

from .schemas import (
    AnnotatedFrameOutput,
    AnnotatedFrameTrackView,
    FrameResolutionMode,
    ListScenesOutput,
    ListTracksOutput,
    ObjectSeedView,
    SceneListItem,
    SceneSummaryOutput,
    SoundTimelineViewOutput,
    TrackSummary,
    VisualEventsOutput,
)

if TYPE_CHECKING:
    from .editor import SoundTimelineEditor

LOW_RES_WIDTH = 640
LOW_RES_JPEG_QUALITY = 75
HIGH_RES_JPEG_QUALITY = 90


class SoundTimelineReadTools:
    def __init__(self, editor: SoundTimelineEditor) -> None:
        self.editor = editor

    def list_scenes(
        self,
        start_scene_index: int = 0,
        limit: int = 25,
    ) -> ListScenesOutput:
        video_asset = self.editor.video_asset
        scenes = video_asset.initial_scenes
        if start_scene_index > len(scenes):
            raise ValueError(f"start_scene_index out of range: {start_scene_index}")
        selected_scenes = scenes[start_scene_index : start_scene_index + limit]
        next_scene_index = start_scene_index + len(selected_scenes)
        if next_scene_index >= len(scenes):
            next_scene_index = None
        return ListScenesOutput(
            frame_count=video_asset.frame_count,
            fps=video_asset.fps,
            total_scene_count=len(scenes),
            start_scene_index=start_scene_index,
            returned_scene_count=len(selected_scenes),
            next_scene_index=next_scene_index,
            scenes=[
                SceneListItem(
                    scene_index=start_scene_index + scene_offset,
                    initial_scene_id=scene.initial_scene_id,
                    start_frame_index=scene.start_frame_index,
                    end_frame_index=scene.end_frame_index,
                    frame_count=scene.frame_count,
                    keyframe_indexes=[
                        keyframe.frame_index for keyframe in scene.keyframes
                    ],
                    track_count=len(scene.scene_tracks),
                    object_seed_count=0
                    if scene.initial_analysis is None
                    else len(scene.initial_analysis.object_seeds),
                )
                for scene_offset, scene in enumerate(selected_scenes)
            ],
        )

    def get_scene_summary(self, scene_index: int) -> SceneSummaryOutput:
        scene = self._scene(scene_index)
        object_seeds: list[ObjectSeedView] = []
        if scene.initial_analysis is not None:
            object_seeds = [
                ObjectSeedView(
                    label=seed.label,
                    tracking_prompt=seed.tracking_prompt,
                    notes=seed.notes,
                )
                for seed in scene.initial_analysis.object_seeds
            ]
        return SceneSummaryOutput(
            scene_index=scene_index,
            initial_scene_id=scene.initial_scene_id,
            start_frame_index=scene.start_frame_index,
            end_frame_index=scene.end_frame_index,
            frame_count=scene.frame_count,
            keyframe_indexes=[keyframe.frame_index for keyframe in scene.keyframes],
            object_seeds=object_seeds,
            tracks=self.list_tracks(scene_index).tracks,
        )

    def get_annotated_frame(
        self,
        frame_index: int,
        resolution_mode: FrameResolutionMode = "low",
    ) -> AnnotatedFrameOutput:
        self.editor.check_frame_index(frame_index, allow_end=False)
        image = extract_frame(Path(self.editor.video_asset.source_path), frame_index)
        tracks = self._tracks_at_frame(frame_index)
        for track_index, track_view in enumerate(tracks):
            color = color_for_index(track_index)
            bbox = track_view.bbox_xyxy
            if bbox is not None:
                draw_bbox(image, bbox, color)
                label_position = (int(bbox[0]), max(0, int(bbox[1]) - 16))
            else:
                label_position = (10, 28 + (track_index * 18))
            draw_label(
                image,
                label_position,
                f"{str(track_view.scene_track_id)[:8]} {track_view.confidence:.2f}",
                color,
            )
        draw_frame_index(image, frame_index)
        image = _render_output_image(image, resolution_mode)
        return AnnotatedFrameOutput(
            frame_index=frame_index,
            resolution_mode=resolution_mode,
            width=image.width,
            height=image.height,
            image=_image_base64(image, quality=_jpeg_quality(resolution_mode)),
            tracks=tracks,
        )

    def list_tracks(self, scene_index: int) -> ListTracksOutput:
        scene = self._scene(scene_index)
        return ListTracksOutput(
            scene_index=scene_index,
            tracks=[
                TrackSummary(
                    scene_track_id=track.scene_track_id,
                    tracking_prompt=track.tracking_prompt,
                    source_label=None
                    if track.source_object_seed is None
                    else track.source_object_seed.label,
                    start_frame_index=track.start_frame_index,
                    end_frame_index=track.end_frame_index,
                    frame_count=track.frame_count,
                    confidence=track.confidence,
                    notes=track.notes,
                )
                for track in scene.scene_tracks
            ],
        )

    def get_visual_events(
        self,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 50,
    ) -> VisualEventsOutput:
        if (
            start_frame_index is not None
            and end_frame_index is not None
            and end_frame_index <= start_frame_index
        ):
            raise ValueError("end_frame_index must be greater than start_frame_index")
        layer = self.editor.video_asset.visual_identity_layer
        if layer is None:
            return VisualEventsOutput(
                total_matching_event_count=0,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                limit=limit,
                returned_event_count=0,
                visual_events=[],
            )
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
            events.append(event)
        events = sorted(
            events,
            key=lambda event: (
                event.start_frame_index,
                event.end_frame_index,
                event.event_type,
                str(event.visual_event_id),
            ),
        )
        paged_events = events[:limit]
        return VisualEventsOutput(
            total_matching_event_count=len(events),
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            limit=limit,
            returned_event_count=len(paged_events),
            visual_events=paged_events,
        )

    def get_sound_timeline(
        self,
        start_frame_index: int | None = None,
        end_frame_index: int | None = None,
        limit: int = 50,
    ) -> SoundTimelineViewOutput:
        if (
            start_frame_index is not None
            and end_frame_index is not None
            and end_frame_index <= start_frame_index
        ):
            raise ValueError("end_frame_index must be greater than start_frame_index")
        timeline = self.editor.video_asset.sound_timeline
        if timeline is None:
            return SoundTimelineViewOutput(
                sound_sources=[],
                sound_events=[],
                total_matching_event_count=0,
                start_frame_index=start_frame_index,
                end_frame_index=end_frame_index,
                limit=limit,
                returned_event_count=0,
            )
        events = []
        for event in timeline.sound_events:
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
            events.append(event)
        events = sorted(
            events,
            key=lambda event: (
                event.start_frame_index,
                event.end_frame_index,
                event.track_type,
                str(event.sound_event_id),
            ),
        )
        paged_events = events[:limit]
        return SoundTimelineViewOutput(
            sound_sources=timeline.sound_sources,
            sound_events=paged_events,
            total_matching_event_count=len(events),
            start_frame_index=start_frame_index,
            end_frame_index=end_frame_index,
            limit=limit,
            returned_event_count=len(paged_events),
            notes=timeline.notes,
        )

    def _scene(self, scene_index: int):
        scenes = self.editor.video_asset.initial_scenes
        if scene_index < 0 or scene_index >= len(scenes):
            raise ValueError(f"scene_index out of range: {scene_index}")
        return scenes[scene_index]

    def _tracks_at_frame(self, frame_index: int) -> list[AnnotatedFrameTrackView]:
        tracks: list[AnnotatedFrameTrackView] = []
        for scene_index, scene in enumerate(self.editor.video_asset.initial_scenes):
            if not (scene.start_frame_index <= frame_index < scene.end_frame_index):
                continue
            for track in scene.scene_tracks:
                for point in track.points:
                    if point.frame_index != frame_index:
                        continue
                    tracks.append(
                        AnnotatedFrameTrackView(
                            scene_index=scene_index,
                            scene_track_id=track.scene_track_id,
                            tracking_prompt=track.tracking_prompt,
                            source_label=None
                            if track.source_object_seed is None
                            else track.source_object_seed.label,
                            bbox_xyxy=point.bbox_xyxy,
                            confidence=point.confidence,
                        )
                    )
        return tracks


def _render_output_image(
    image: Image.Image,
    resolution_mode: FrameResolutionMode,
) -> Image.Image:
    if resolution_mode == "high":
        return image

    if image.width <= LOW_RES_WIDTH:
        return image

    height = round(image.height * (LOW_RES_WIDTH / image.width))
    return image.resize((LOW_RES_WIDTH, height), Image.Resampling.LANCZOS)


def _jpeg_quality(resolution_mode: FrameResolutionMode) -> int:
    if resolution_mode == "high":
        return HIGH_RES_JPEG_QUALITY
    return LOW_RES_JPEG_QUALITY


def _image_base64(image: Image.Image, *, quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
