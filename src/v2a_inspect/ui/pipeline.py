from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from v2a_inspect.agents.sound_timeline import run_sound_timeline_agent
from v2a_inspect.client import SAM3Client, VideoClient
from v2a_inspect.models import VideoAsset
from v2a_inspect.preprocessing import (
    analyze_initial_scene,
    build_visual_identity_layer,
    compute_visual_events,
    detect_initial_scenes,
    extract_keyframes_for_initial_scene,
    prepare_video,
    sam3_tracking_video_path,
    track_initial_scenes_object_seeds,
)

from .store import VideoAssetStore


@dataclass(frozen=True)
class PipelineOptions:
    scene_threshold: float = 27.0
    max_keyframes_per_scene: int = 20
    server_url: str | None = None


async def run_uploaded_video_pipeline(
    upload_path: Path,
    work_dir: Path,
    store: VideoAssetStore,
    options: PipelineOptions,
) -> None:
    """Run the full pipeline and publish each completed VideoAsset stage."""

    try:
        loop = asyncio.get_running_loop()
        await store.set_running(stage="prepare video")
        video_asset = await asyncio.to_thread(prepare_video, upload_path, work_dir)
        await store.set_asset(video_asset, stage="prepared video")

        await store.touch(stage="detect scenes")
        scenes = await asyncio.to_thread(
            detect_initial_scenes,
            video_asset,
            threshold=options.scene_threshold,
        )
        video_asset = video_asset.model_copy(update={"initial_scenes": scenes})
        await store.publish_asset_mutation(video_asset, stage="detected scenes")

        await store.touch(stage="extract keyframes")
        video_asset = await _extract_keyframes_incrementally(
            video_asset,
            work_dir,
            store,
            max_keyframes_per_scene=options.max_keyframes_per_scene,
        )
        await store.publish_asset_mutation(video_asset, stage="extracted keyframes")

        await store.touch(stage="analyze scenes")
        video_asset = await _analyze_scenes_incrementally(
            video_asset,
            store,
        )
        await store.publish_asset_mutation(video_asset, stage="analyzed scenes")

        await store.touch(stage="track objects")
        video_asset = await _track_objects(video_asset, options.server_url, store)
        await store.publish_asset_mutation(video_asset, stage="tracked objects")

        await store.touch(stage="build visual identity")
        video_asset = await asyncio.to_thread(build_visual_identity_layer, video_asset)
        await store.publish_asset_mutation(video_asset, stage="built visual identity")

        await store.touch(stage="compute visual events")
        video_asset = await asyncio.to_thread(compute_visual_events, video_asset)
        await store.publish_asset_mutation(video_asset, stage="computed visual events")

        await store.touch(stage="build sound timeline")
        on_sound_change = _threadsafe_publish_callback(loop, store, video_asset)
        await asyncio.to_thread(
            run_sound_timeline_agent,
            video_asset,
            on_change=on_sound_change,
        )
        await store.publish_asset_mutation(video_asset, stage="built sound timeline")

        await store.set_complete()
    except Exception as exc:  # noqa: BLE001 - UI needs stage-specific failure text.
        await store.set_error(str(exc))


async def _extract_keyframes_incrementally(
    video_asset: VideoAsset,
    work_dir: Path,
    store: VideoAssetStore,
    *,
    max_keyframes_per_scene: int,
) -> VideoAsset:
    updated_scenes = list(video_asset.initial_scenes)
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        keyframes = await asyncio.to_thread(
            extract_keyframes_for_initial_scene,
            video_asset,
            scene,
            work_dir,
            max_keyframes_per_scene,
        )
        updated_scenes[scene_index] = scene.model_copy(update={"keyframes": keyframes})
        video_asset = video_asset.model_copy(update={"initial_scenes": updated_scenes})
        await store.publish_asset_mutation(
            video_asset,
            stage=f"extracted keyframes {scene_index + 1}/{len(updated_scenes)}",
        )
    return video_asset


async def _analyze_scenes_incrementally(
    video_asset: VideoAsset,
    store: VideoAssetStore,
) -> VideoAsset:
    updated_scenes = list(video_asset.initial_scenes)
    for scene_index, scene in enumerate(video_asset.initial_scenes):
        updated_scenes[scene_index] = await asyncio.to_thread(
            analyze_initial_scene,
            scene,
        )
        video_asset = video_asset.model_copy(update={"initial_scenes": updated_scenes})
        await store.publish_asset_mutation(
            video_asset,
            stage=f"analyzed scenes {scene_index + 1}/{len(updated_scenes)}",
        )
    return video_asset


async def _track_objects(
    video_asset: VideoAsset,
    server_url: str | None,
    store: VideoAssetStore,
) -> VideoAsset:
    tracking_video_path = sam3_tracking_video_path(video_asset)
    async with VideoClient(base_url=server_url) as video_client:
        upload_response = await video_client.upload(str(tracking_video_path))
    async with SAM3Client(base_url=server_url) as sam_client:
        for scene_index in range(len(video_asset.initial_scenes)):
            video_asset = await track_initial_scenes_object_seeds(
                video_asset,
                video_id=upload_response.video_id,
                sam_client=sam_client,
                scene_indexes=[scene_index],
            )
            video_asset = await _rebuild_visual_layers(video_asset)
            await store.publish_asset_mutation(
                video_asset,
                stage=f"tracked objects {scene_index + 1}/{len(video_asset.initial_scenes)}",
            )
    return video_asset


async def _rebuild_visual_layers(video_asset: VideoAsset) -> VideoAsset:
    video_asset = await asyncio.to_thread(build_visual_identity_layer, video_asset)
    return await asyncio.to_thread(compute_visual_events, video_asset)


def _threadsafe_publish_callback(
    loop: asyncio.AbstractEventLoop,
    store: VideoAssetStore,
    video_asset: VideoAsset,
) -> Callable[[str], None]:
    def publish(stage: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            store.publish_asset_mutation(video_asset, stage=stage),
            loop,
        )
        future.result()

    return publish
