from __future__ import annotations

import unittest

from v2a_inspect.contracts import (
    AmbienceBed,
    EvidenceWindow,
    GenerationGroup,
    LabelCandidate,
    PhysicalSourceTrack,
    RoutingDecision,
    SoundEventSegment,
)
from v2a_inspect.tools.types import VideoProbe
from v2a_inspect.workflows import InspectOptions
from v2a_inspect_server.finalize import build_final_bundle, finalize_route_decision


class FinalizeTests(unittest.TestCase):
    def test_finalize_route_prefers_tta_for_ambience(self) -> None:
        group = GenerationGroup(
            group_id="gen-0001",
            member_ambience_ids=["ambience-0001"],
            canonical_label="ambience:scene_bed",
            canonical_description="placeholder",
            group_confidence=0.7,
            route_decision=RoutingDecision(model_type="VTA", confidence=0.5, factors=[], reasoning="", rule_based=True),
        )
        decision = finalize_route_decision(group)
        self.assertEqual(decision.model_type, "TTA")

    def test_build_final_bundle_persists_validation(self) -> None:
        state = {
            "video_path": "/tmp/video.mp4",
            "video_probe": VideoProbe(video_path="/tmp/video.mp4", duration_seconds=3.0, fps=2.0, width=320, height=240),
            "candidate_cuts": [],
            "evidence_windows": [EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=3.0)],
            "physical_sources": [
                PhysicalSourceTrack(source_id="source-0000", kind="foreground", label_candidates=[LabelCandidate(label="person", score=0.9)], spans=[(0.0, 1.0)], track_refs=["trk0"], identity_confidence=0.4, reid_neighbors=[])
            ],
            "sound_event_segments": [
                SoundEventSegment(event_id="event-0000", source_id="source-0000", start_time=0.0, end_time=1.0, event_type="continuous_motion", confidence=0.8)
            ],
            "ambience_beds": [
                AmbienceBed(ambience_id="ambience-0000", start_time=0.0, end_time=3.0, environment_type="scene_bed", acoustic_profile="continuous visual environment texture", confidence=0.7)
            ],
            "generation_groups": [
                GenerationGroup(group_id="gen-0000", member_event_ids=["event-0000"], canonical_label="person:continuous_motion:ground_contact", canonical_description="placeholder", group_confidence=0.8, route_decision=RoutingDecision(model_type="VTA", confidence=0.6, factors=[], reasoning="", rule_based=True))
            ],
            "storyboard_path": "/tmp/storyboard/storyboard.jpg",
            "track_crops": [],
        }
        bundle = build_final_bundle(state)
        self.assertGreaterEqual(len(bundle.validation.issues), 1)
        self.assertEqual(bundle.generation_groups[0].canonical_description.startswith("person"), True)

    def test_build_final_bundle_uses_description_writer_when_available(self) -> None:
        class _FakeWriter:
            def write_group_description(self, context: dict[str, object]) -> object:
                return type(
                    "Draft",
                    (),
                    {
                        "canonical_description": "rich inferred footsteps",
                        "description_confidence": 0.91,
                        "description_rationale": f"writer saw {context['canonical_label']}",
                    },
                )()

        state = {
            "video_path": "/tmp/video.mp4",
            "video_probe": VideoProbe(video_path="/tmp/video.mp4", duration_seconds=3.0, fps=2.0, width=320, height=240),
            "candidate_cuts": [],
            "evidence_windows": [EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=3.0)],
            "physical_sources": [
                PhysicalSourceTrack(source_id="source-0000", kind="foreground", label_candidates=[LabelCandidate(label="person", score=0.9)], spans=[(0.0, 1.0)], track_refs=["trk0"], identity_confidence=0.8, reid_neighbors=[])
            ],
            "sound_event_segments": [
                SoundEventSegment(event_id="event-0000", source_id="source-0000", start_time=0.0, end_time=1.0, event_type="continuous_motion", confidence=0.8)
            ],
            "ambience_beds": [],
            "generation_groups": [
                GenerationGroup(group_id="gen-0000", member_event_ids=["event-0000"], canonical_label="person:continuous_motion:ground_contact", canonical_description="placeholder", group_confidence=0.8, route_decision=RoutingDecision(model_type="VTA", confidence=0.6, factors=[], reasoning="", rule_based=True))
            ],
            "storyboard_path": "/tmp/storyboard/storyboard.jpg",
            "track_crops": [],
        }
        bundle = build_final_bundle(state, description_writer=_FakeWriter())
        self.assertEqual(bundle.generation_groups[0].canonical_description, "rich inferred footsteps")
        self.assertEqual(bundle.generation_groups[0].description_rationale, "writer saw person:continuous_motion:ground_contact")

    def test_build_final_bundle_records_active_pipeline_mode_and_recovery_actions(self) -> None:
        state = {
            "video_path": "/tmp/video.mp4",
            "options": InspectOptions(pipeline_mode="agentic_tool_first"),
            "video_probe": VideoProbe(video_path="/tmp/video.mp4", duration_seconds=3.0, fps=2.0, width=320, height=240),
            "candidate_cuts": [],
            "evidence_windows": [EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=3.0)],
            "physical_sources": [],
            "sound_event_segments": [],
            "ambience_beds": [],
            "generation_groups": [],
            "storyboard_path": "/tmp/storyboard/storyboard.jpg",
            "track_crops": [],
            "frames_per_window": 6,
            "recovery_actions": ["densify_window_sampling", "recover_foreground_sources"],
            "recovery_attempts": [{"tool_name": "densify_window_sampling", "frames_per_scene": 6}],
            "stage_history": [{"kind": "stage", "stage": "extract_entities", "elapsed_seconds": 1.23}],
            "runtime_trace_path": "/tmp/run/video-runtime-trace.jsonl",
        }
        bundle = build_final_bundle(state)
        self.assertEqual(bundle.pipeline_metadata["pipeline_version"], "agentic_tool_first")
        self.assertEqual(bundle.pipeline_metadata["sampling_frames_per_window"], 6)
        self.assertEqual(
            bundle.pipeline_metadata["recovery_actions"],
            ["densify_window_sampling", "recover_foreground_sources"],
        )
        self.assertEqual(
            bundle.pipeline_metadata["recovery_attempts"],
            [{"tool_name": "densify_window_sampling", "frames_per_scene": 6}],
        )
        self.assertEqual(
            bundle.pipeline_metadata["stage_history"],
            [{"kind": "stage", "stage": "extract_entities", "elapsed_seconds": 1.23}],
        )
        self.assertEqual(
            bundle.pipeline_metadata["runtime_trace_path"],
            "/tmp/run/video-runtime-trace.jsonl",
        )
