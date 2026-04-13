from __future__ import annotations

import unittest

from v2a_inspect.contracts import EvidenceWindow, LabelCandidate, PhysicalSourceTrack
from v2a_inspect.tools.types import Sam3EntityTrack, Sam3VisualFeatures
from v2a_inspect_server.semantics import (
    build_ambience_beds,
    build_generation_groups,
    group_acoustic_recipes,
    build_sound_event_segments,
)


class SemanticLayerTests(unittest.TestCase):
    def test_one_source_can_split_into_multiple_event_segments(self) -> None:
        source = PhysicalSourceTrack(
            source_id="source-0000",
            kind="foreground",
            label_candidates=[LabelCandidate(label="person", score=0.9)],
            spans=[(0.0, 1.0), (2.0, 3.0)],
            track_refs=["trk0", "trk1"],
            identity_confidence=0.9,
            reid_neighbors=[],
        )
        tracks_by_id = {
            "trk0": Sam3EntityTrack(
                track_id="trk0",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.9,
                features=Sam3VisualFeatures(motion_score=0.8, interaction_score=0.7),
            ),
            "trk1": Sam3EntityTrack(
                track_id="trk1",
                scene_index=1,
                start_seconds=2.0,
                end_seconds=3.0,
                confidence=0.9,
                features=Sam3VisualFeatures(motion_score=0.1, interaction_score=0.0),
            ),
        }
        events = build_sound_event_segments([source], tracks_by_id=tracks_by_id)
        self.assertEqual(len(events), 2)
        self.assertNotEqual(events[0].event_type, events[1].event_type)

    def test_different_sources_can_share_one_generation_group(self) -> None:
        sources = [
            PhysicalSourceTrack(
                source_id="source-0000",
                kind="foreground",
                label_candidates=[LabelCandidate(label="person", score=0.9)],
                spans=[(0.0, 1.0)],
                track_refs=["trk0"],
                identity_confidence=0.9,
                reid_neighbors=[],
            ),
            PhysicalSourceTrack(
                source_id="source-0001",
                kind="foreground",
                label_candidates=[LabelCandidate(label="person", score=0.9)],
                spans=[(2.0, 3.0)],
                track_refs=["trk1"],
                identity_confidence=0.9,
                reid_neighbors=[],
            ),
        ]
        tracks_by_id = {
            "trk0": Sam3EntityTrack(
                track_id="trk0",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.9,
                features=Sam3VisualFeatures(motion_score=0.8),
            ),
            "trk1": Sam3EntityTrack(
                track_id="trk1",
                scene_index=1,
                start_seconds=2.0,
                end_seconds=3.0,
                confidence=0.9,
                features=Sam3VisualFeatures(motion_score=0.8),
            ),
        }
        events = build_sound_event_segments(sources, tracks_by_id=tracks_by_id)
        groups = build_generation_groups(events, [], physical_sources=sources)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].member_event_ids), 2)

    def test_group_acoustic_recipes_exposes_recipe_signatures(self) -> None:
        sources = [
            PhysicalSourceTrack(
                source_id="source-0000",
                kind="foreground",
                label_candidates=[LabelCandidate(label="cymbal", score=0.95)],
                spans=[(0.0, 1.0)],
                track_refs=["scene-0-track-0"],
                identity_confidence=0.9,
                reid_neighbors=[],
            )
        ]
        tracks_by_id = {
            "scene-0-track-0": Sam3EntityTrack(
                track_id="scene-0-track-0",
                scene_index=0,
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.9,
                features=Sam3VisualFeatures(motion_score=0.8, interaction_score=0.8),
            ),
        }
        events = build_sound_event_segments(sources, tracks_by_id=tracks_by_id)
        groups, recipe_signatures = group_acoustic_recipes(
            events,
            [],
            physical_sources=sources,
            scene_hypotheses_by_window={0: {"material_cues": ["metal"], "interactions": ["striking"]}},
            proposal_provenance_by_window={0: {"ontology_semantics": ["metal", "striking"]}},
        )
        self.assertEqual(len(groups), 1)
        self.assertIn("gen-0000", recipe_signatures)
        self.assertEqual(recipe_signatures["gen-0000"].source_label, "cymbal")

    def test_ambience_beds_are_explicit_when_windows_lack_foreground_coverage(
        self,
    ) -> None:
        windows = [
            EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=3.0)
        ]
        sources = [
            PhysicalSourceTrack(
                source_id="source-0000",
                kind="foreground",
                label_candidates=[],
                spans=[(0.0, 0.5)],
                track_refs=["trk0"],
                identity_confidence=0.8,
                reid_neighbors=[],
            )
        ]
        ambience = build_ambience_beds(windows, sources)
        self.assertEqual(len(ambience), 1)
        self.assertEqual(ambience[0].environment_type, "scene_bed")


if __name__ == "__main__":
    unittest.main()
