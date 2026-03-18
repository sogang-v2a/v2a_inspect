from __future__ import annotations

from v2a_inspect.workflows.state import InspectState

from ..response_models import GroupedAnalysis
from ._shared import append_state_message, get_active_groups


def assemble_grouped_analysis(state: InspectState) -> dict[str, object]:
    """Build the final grouped analysis and annotate the copied scene analysis."""

    scene_analysis = state.get("scene_analysis")
    if scene_analysis is None:
        raise ValueError(
            "assemble_grouped_analysis requires 'scene_analysis' in state."
        )

    raw_tracks = list(state.get("raw_tracks", []))
    groups = [group.model_copy(deep=True) for group in get_active_groups(state)]

    track_to_group: dict[str, str] = {}
    track_to_canonical: dict[str, str] = {}
    for group in groups:
        for track_id in group.member_ids:
            track_to_group[track_id] = group.group_id
            track_to_canonical[track_id] = group.canonical_description

    annotated_scene_analysis = scene_analysis.model_copy(deep=True)
    for scene in annotated_scene_analysis.scenes:
        scene_index = scene.scene_index
        background_track_id = f"s{scene_index}_bg"
        scene.background_group_id = track_to_group.get(background_track_id)
        scene.background_canonical = track_to_canonical.get(background_track_id)

        for object_index, obj in enumerate(scene.objects):
            track_id = f"s{scene_index}_obj{object_index}"
            obj.group_id = track_to_group.get(track_id)
            obj.canonical_description = track_to_canonical.get(track_id)

    grouped_analysis = GroupedAnalysis(
        scene_analysis=annotated_scene_analysis,
        raw_tracks=raw_tracks,
        groups=groups,
        track_to_group=track_to_group,
    )
    return {
        "grouped_analysis": grouped_analysis,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Assembled grouped analysis with {len(groups)} groups.",
        ),
    }
