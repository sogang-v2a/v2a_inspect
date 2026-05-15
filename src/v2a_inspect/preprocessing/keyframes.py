from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from v2a_inspect.media_utils import extract_frame
from v2a_inspect.models import InitialScene, Keyframe, VideoAsset

KEYFRAME_INTERVAL_FRAMES = 60


def select_initial_scene_keyframe_indexes(
    scene: InitialScene,
    max_keyframes: int = 20,
) -> list[int]:
    """Select deterministic representative frame indexes for one initial scene."""

    if max_keyframes < 1:
        raise ValueError("max_keyframes must be greater than or equal to 1")
    if scene.frame_count <= 0:
        return []

    keyframe_count = _keyframe_count(scene.frame_count, max_keyframes)

    selected: list[int] = []
    for index in range(1, keyframe_count + 1):
        frame_index = scene.start_frame_index + round(
            scene.frame_count * index / (keyframe_count + 1)
        )
        clamped_frame_index = max(
            scene.start_frame_index,
            min(frame_index, scene.end_frame_index - 1),
        )
        if clamped_frame_index not in selected:
            selected.append(clamped_frame_index)

    return selected


def extract_keyframes_for_initial_scene(
    video_asset: VideoAsset,
    scene: InitialScene,
    work_dir: Path,
    max_keyframes: int = 20,
) -> list[Keyframe]:
    """Extract and save selected keyframes for one initial scene."""

    keyframes: list[Keyframe] = []
    scene_keyframe_dir = (
        work_dir / "keyframes" / "initial_scenes" / str(scene.initial_scene_id)
    )
    scene_keyframe_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in select_initial_scene_keyframe_indexes(scene, max_keyframes):
        keyframe_id = uuid4()
        image_path = _keyframe_image_path(
            scene_keyframe_dir,
            frame_index,
            keyframe_id,
        )
        keyframe = Keyframe(
            keyframe_id=keyframe_id,
            frame_index=frame_index,
            image_path=image_path,
        )
        image = extract_frame(video_asset.source_path, frame_index)
        image.save(keyframe.image_path, format="JPEG")
        keyframes.append(keyframe)

    return keyframes


def extract_keyframes_for_initial_scenes(
    video_asset: VideoAsset,
    work_dir: Path,
    max_keyframes_per_scene: int = 20,
) -> list[InitialScene]:
    """Return initial scenes with freshly extracted keyframes attached."""

    updated_scenes: list[InitialScene] = []
    for scene in video_asset.initial_scenes:
        keyframes = extract_keyframes_for_initial_scene(
            video_asset,
            scene,
            work_dir,
            max_keyframes_per_scene,
        )
        updated_scenes.append(scene.model_copy(update={"keyframes": keyframes}))

    return updated_scenes


def _keyframe_count(frame_count: int, max_keyframes: int) -> int:
    keyframe_count = frame_count // KEYFRAME_INTERVAL_FRAMES
    if frame_count % KEYFRAME_INTERVAL_FRAMES > 0:
        keyframe_count += 1

    if keyframe_count < 1:
        keyframe_count = 1

    if keyframe_count > max_keyframes:
        keyframe_count = max_keyframes

    return keyframe_count


def _keyframe_image_path(
    keyframe_dir: Path, frame_index: int, keyframe_id: UUID
) -> Path:
    return keyframe_dir / f"{frame_index:08d}_{keyframe_id}.jpg"
