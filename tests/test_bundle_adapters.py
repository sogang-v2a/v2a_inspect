from __future__ import annotations

import unittest

from v2a_inspect.contracts import (
    ArtifactRefs,
    AmbienceBed,
    EvidenceWindow,
    GenerationGroup,
    MultitrackDescriptionBundle,
    RoutingDecision,
    SoundEventSegment,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.contracts.adapters import bundle_to_grouped_analysis


class BundleAdapterTests(unittest.TestCase):
    def test_bundle_to_grouped_analysis_preserves_generation_group_membership(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="vid-001",
            video_meta=VideoMeta(duration_seconds=3.0, fps=2.0, width=320, height=240),
            evidence_windows=[EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=3.0)],
            sound_events=[
                SoundEventSegment(
                    event_id="event-0000",
                    source_id="source-0000",
                    start_time=0.0,
                    end_time=1.0,
                    event_type="continuous_motion",
                    confidence=0.9,
                )
            ],
            ambience_beds=[
                AmbienceBed(
                    ambience_id="ambience-0000",
                    start_time=0.0,
                    end_time=3.0,
                    environment_type="scene_bed",
                    acoustic_profile="continuous visual environment texture",
                    confidence=0.7,
                )
            ],
            generation_groups=[
                GenerationGroup(
                    group_id="gen-0000",
                    member_event_ids=["event-0000"],
                    canonical_label="person:continuous_motion:ground_contact",
                    canonical_description="running footsteps",
                    group_confidence=0.9,
                    route_decision=RoutingDecision(model_type="VTA", confidence=0.8, factors=["continuous_motion"], reasoning="heuristic", rule_based=True),
                ),
                GenerationGroup(
                    group_id="gen-0001",
                    member_ambience_ids=["ambience-0000"],
                    canonical_label="ambience:scene_bed",
                    canonical_description="continuous visual environment texture",
                    group_confidence=0.7,
                    route_decision=RoutingDecision(model_type="VTA", confidence=0.9, factors=["ambience_bed"], reasoning="heuristic", rule_based=True),
                ),
            ],
            validation=ValidationReport(status="pass_with_warnings"),
            artifacts=ArtifactRefs(storyboard_path="/tmp/storyboard.jpg"),
        )
        grouped = bundle_to_grouped_analysis(bundle)
        self.assertEqual(len(grouped.raw_tracks), 2)
        self.assertEqual(len(grouped.groups), 2)
        self.assertEqual(grouped.track_to_group["event-0000"], "gen-0000")
        self.assertEqual(grouped.track_to_group["ambience-0000"], "gen-0001")


if __name__ == "__main__":
    unittest.main()
