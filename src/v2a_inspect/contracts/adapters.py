from __future__ import annotations

from collections import defaultdict

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.pipeline.response_models import (
    GroupedAnalysis,
    RawTrack,
    Scene,
    TimeRange,
    TrackGroup,
    VideoSceneAnalysis,
)


def bundle_to_grouped_analysis(
    bundle: MultitrackDescriptionBundle,
    *,
    scene_analysis: VideoSceneAnalysis | None = None,
) -> GroupedAnalysis:
    resolved_scene_analysis = scene_analysis or _scene_analysis_from_bundle(bundle)
    raw_tracks: list[RawTrack] = []
    track_to_group: dict[str, str] = {}
    groups_by_id: dict[str, TrackGroup] = {}

    event_group_lookup = {
        event_id: group.group_id
        for group in bundle.generation_groups
        for event_id in group.member_event_ids
    }
    ambience_group_lookup = {
        ambience_id: group.group_id
        for group in bundle.generation_groups
        for ambience_id in group.member_ambience_ids
    }

    for event in bundle.sound_events:
        track_id = event.event_id
        group_id = event_group_lookup.get(event.event_id, "ungrouped")
        raw_tracks.append(
            RawTrack(
                track_id=track_id,
                scene_index=_scene_index_for_time(bundle, event.start_time),
                kind="object",
                description=event.event_type,
                start=event.start_time,
                end=event.end_time,
                obj_index=None,
                n_scene_objects=0,
            )
        )
        track_to_group[track_id] = group_id

    for ambience in bundle.ambience_beds:
        track_id = ambience.ambience_id
        group_id = ambience_group_lookup.get(ambience.ambience_id, "ungrouped")
        raw_tracks.append(
            RawTrack(
                track_id=track_id,
                scene_index=_scene_index_for_time(bundle, ambience.start_time),
                kind="background",
                description=ambience.acoustic_profile,
                start=ambience.start_time,
                end=ambience.end_time,
                obj_index=None,
                n_scene_objects=0,
            )
        )
        track_to_group[track_id] = group_id

    members_by_group: dict[str, list[str]] = defaultdict(list)
    for track_id, group_id in track_to_group.items():
        members_by_group[group_id].append(track_id)

    for group in bundle.generation_groups:
        groups_by_id[group.group_id] = TrackGroup(
            group_id=group.group_id,
            canonical_description=group.canonical_description,
            member_ids=members_by_group.get(group.group_id, []),
            vlm_verified=False,
        )

    return GroupedAnalysis(
        scene_analysis=resolved_scene_analysis,
        raw_tracks=raw_tracks,
        groups=list(groups_by_id.values()),
        track_to_group=track_to_group,
    )


def _scene_analysis_from_bundle(bundle: MultitrackDescriptionBundle) -> VideoSceneAnalysis:
    scenes = [
        Scene(
            scene_index=index,
            time_range=TimeRange(start=window.start_time, end=window.end_time),
            background_sound="",
            objects=[],
        )
        for index, window in enumerate(bundle.evidence_windows)
    ]
    return VideoSceneAnalysis(total_duration=bundle.video_meta.duration_seconds, scenes=scenes)


def _scene_index_for_time(bundle: MultitrackDescriptionBundle, timestamp: float) -> int:
    for index, window in enumerate(bundle.evidence_windows):
        if window.start_time <= timestamp <= window.end_time:
            return index
    return 0
