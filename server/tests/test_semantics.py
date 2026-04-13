from __future__ import annotations

import unittest

from v2a_inspect.contracts import AmbienceBed, LabelCandidate, PhysicalSourceTrack, TrackCrop
from v2a_inspect.tools.types import Sam3EntityTrack, Sam3VisualFeatures
from v2a_inspect_server.gemini_grouping import group_generation_groups
from v2a_inspect_server.gemini_source_semantics import build_source_and_event_semantics


class SemanticLayerTests(unittest.TestCase):
    def test_build_source_and_event_semantics_emits_one_unresolved_event_per_span(self) -> None:
        source = PhysicalSourceTrack(
            source_id="source-0000",
            kind="foreground",
            label_candidates=[LabelCandidate(label="paddle", score=0.9)],
            spans=[(0.0, 1.0), (2.0, 3.0)],
            track_refs=["trk0"],
            crop_refs=["crop-0000"],
            window_refs=["window-0000"],
            identity_confidence=0.9,
            reid_neighbors=[],
        )
        outputs = build_source_and_event_semantics(
            physical_sources=[source],
            tracks_by_id={
                "trk0": Sam3EntityTrack(
                    track_id="trk0",
                    scene_index=0,
                    start_seconds=0.0,
                    end_seconds=3.0,
                    confidence=0.8,
                    features=Sam3VisualFeatures(motion_score=0.5),
                )
            },
            track_crops=[
                TrackCrop(
                    crop_id="crop-0000",
                    track_id="trk0",
                    scene_index=0,
                    frame_path="/tmp/frame.jpg",
                    crop_path="/tmp/crop.jpg",
                    timestamp_seconds=0.5,
                    bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
                )
            ],
            evidence_windows=[],
            interpreter=None,
        )
        self.assertEqual(len(outputs["sound_events"]), 2)
        self.assertEqual([event.event_type for event in outputs["sound_events"]], ["", ""])
        self.assertEqual(outputs["ambience_beds"], [])

    def test_group_generation_groups_defaults_to_singletons_without_judge(self) -> None:
        source = PhysicalSourceTrack(
            source_id="source-0000",
            kind="foreground",
            label_candidates=[LabelCandidate(label="paddle", score=0.9)],
            spans=[(0.0, 1.0)],
            track_refs=["trk0"],
            identity_confidence=0.9,
            reid_neighbors=[],
        )
        groups = group_generation_groups(
            sound_events=[],
            ambience_beds=[
                AmbienceBed(
                    ambience_id="ambience-0000",
                    start_time=0.0,
                    end_time=3.0,
                    environment_type="room",
                    acoustic_profile="",
                    confidence=0.4,
                )
            ],
            physical_sources=[source],
            grouping_judge=None,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].member_ambience_ids, ["ambience-0000"])


if __name__ == "__main__":
    unittest.main()
