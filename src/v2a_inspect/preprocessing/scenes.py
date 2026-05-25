from __future__ import annotations

from typing import Protocol

from scenedetect import ContentDetector, SceneManager, open_video

from v2a_inspect.models import InitialScene, VideoAsset


class _FrameTimecode(Protocol):
    def get_frames(self) -> int: ...


def detect_initial_scenes(
    video_asset: VideoAsset,
    threshold: float = 27.0,
) -> list[InitialScene]:
    """Detect coarse scene boundaries with PySceneDetect."""

    video = open_video(str(video_asset.source_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scenes: list[InitialScene] = []
    for start_timecode, end_timecode in scene_manager.get_scene_list():
        start = _clamp_frame_index(start_timecode, video_asset.frame_count)
        end = _clamp_frame_index(end_timecode, video_asset.frame_count)
        if start >= end:
            continue

        scenes.append(InitialScene(start_frame_index=start, end_frame_index=end))

    if not scenes:
        return [_full_video_scene(video_asset)]

    scenes[0].start_frame_index = 0
    scenes[-1].end_frame_index = video_asset.frame_count
    return [
        scene for scene in scenes if scene.start_frame_index < scene.end_frame_index
    ]


def _clamp_frame_index(timecode: _FrameTimecode, frame_count: int) -> int:
    return max(0, min(timecode.get_frames(), frame_count))


def _full_video_scene(video_asset: VideoAsset) -> InitialScene:
    return InitialScene(start_frame_index=0, end_frame_index=video_asset.frame_count)
