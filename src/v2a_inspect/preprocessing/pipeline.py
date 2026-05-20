from __future__ import annotations

from pathlib import Path

from v2a_inspect.models import VideoAsset

from .keyframes import extract_keyframes_for_initial_scenes
from .scenes import detect_initial_scenes
from .video import prepare_video


def preprocess_video(
    raw_video_path: Path,
    work_dir: Path,
    scene_threshold: float = 27.0,
    max_keyframes_per_scene: int = 20,
) -> VideoAsset:
    """Run the local preprocessing pipeline and return a populated VideoAsset."""

    work_dir.mkdir(parents=True, exist_ok=True)

    video_asset = prepare_video(raw_video_path, work_dir)
    initial_scenes = detect_initial_scenes(video_asset, threshold=scene_threshold)
    video_asset = video_asset.model_copy(update={"initial_scenes": initial_scenes})

    initial_scenes = extract_keyframes_for_initial_scenes(
        video_asset,
        work_dir,
        max_keyframes_per_scene=max_keyframes_per_scene,
    )
    return video_asset.model_copy(update={"initial_scenes": initial_scenes})
