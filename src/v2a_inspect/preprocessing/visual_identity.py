from __future__ import annotations

from v2a_inspect.models import (
    SceneTrack,
    VideoAsset,
    VisualIdentityLayer,
    VisualObject,
    VisualPresence,
)


def build_visual_identity_layer(video_asset: VideoAsset) -> VideoAsset:
    """Rebuild the visual identity layer from current scene tracks."""

    visual_objects: list[VisualObject] = []
    for initial_scene in video_asset.initial_scenes:
        for scene_track in initial_scene.scene_tracks:
            visual_objects.append(_visual_object_from_scene_track(scene_track))

    visual_identity_layer = VisualIdentityLayer(visual_objects=visual_objects)
    return video_asset.model_copy(
        update={"visual_identity_layer": visual_identity_layer}
    )


def _visual_object_from_scene_track(scene_track: SceneTrack) -> VisualObject:
    presence = VisualPresence(
        start_frame_index=scene_track.start_frame_index,
        end_frame_index=scene_track.end_frame_index,
        state="visible",
        scene_track_id=scene_track.scene_track_id,
    )
    return VisualObject(
        label=_scene_track_label(scene_track),
        presences=[presence],
    )


def _scene_track_label(scene_track: SceneTrack) -> str | None:
    if scene_track.source_object_seed is not None:
        return scene_track.source_object_seed.label
    return scene_track.tracking_prompt
