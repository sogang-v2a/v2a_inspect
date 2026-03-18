from __future__ import annotations

from v2a_inspect.workflows.state import InspectState

from ..response_models import RawTrack
from ._shared import append_state_message


def extract_raw_tracks(state: InspectState) -> dict[str, object]:
    """Flatten scene analysis output into ordered raw tracks."""

    scene_analysis = state.get("scene_analysis")
    if scene_analysis is None:
        raise ValueError("extract_raw_tracks requires 'scene_analysis' in state.")

    tracks: list[RawTrack] = []
    for scene in scene_analysis.scenes:
        scene_index = scene.scene_index
        n_objects = len(scene.objects)
        tracks.append(
            RawTrack(
                track_id=f"s{scene_index}_bg",
                scene_index=scene_index,
                kind="background",
                description=scene.background_sound,
                start=scene.time_range.start,
                end=scene.time_range.end,
                obj_index=None,
                n_scene_objects=n_objects,
            )
        )
        for object_index, obj in enumerate(scene.objects):
            tracks.append(
                RawTrack(
                    track_id=f"s{scene_index}_obj{object_index}",
                    scene_index=scene_index,
                    kind="object",
                    description=obj.description,
                    start=obj.time_range.start,
                    end=obj.time_range.end,
                    obj_index=object_index,
                    n_scene_objects=n_objects,
                )
            )

    return {
        "raw_tracks": tracks,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Extracted {len(tracks)} raw tracks.",
        ),
    }
