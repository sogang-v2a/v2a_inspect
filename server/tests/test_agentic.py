from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from v2a_inspect.agent import AgentIssue
from v2a_inspect.contracts import (
    ArtifactRefs,
    EvidenceWindow,
    LabelCandidate,
    MultitrackDescriptionBundle,
    PhysicalSourceTrack,
    SoundEventSegment,
    ValidationIssue,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.tools.types import FrameBatch, SampledFrame, Sam3EntityTrack, Sam3TrackPoint, Sam3TrackSet
from server.tests.fakes import build_fake_tooling_runtime
from v2a_inspect_server.agentic import _adjudicate_issue, _build_issues, run_agent_review_pass, run_agentic_tool_loop


class AgenticIntegrationTests(unittest.TestCase):
    def test_agent_review_pass_logs_bounded_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            bundle = MultitrackDescriptionBundle(
                video_id="video",
                video_meta=VideoMeta(duration_seconds=1.0, fps=2.0, width=320, height=240),
                evidence_windows=[],
                validation=ValidationReport(
                    status="pass_with_warnings",
                    issues=[
                        ValidationIssue(
                            issue_id="route-0000",
                            issue_type="route_inconsistency",
                            severity="warning",
                            message="route mismatch",
                        )
                    ],
                ),
                artifacts=ArtifactRefs(run_dir=str(run_dir), storyboard_path=str(run_dir / "storyboard.jpg")),
            )
            bundle.evidence_windows = [SimpleNamespace(window_id="window-0000", start_time=0.0, end_time=1.0)]
            inspect_state = {
                "video_path": "/tmp/video.mp4",
                "frame_batches": [FrameBatch(scene_index=0, frames=[SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path="/tmp/frame.jpg")])],
                "sam3_track_set": SimpleNamespace(tracks=[]),
            }
            tooling_runtime = SimpleNamespace(
                sam3_client=SimpleNamespace(
                    extract_entities=lambda frame_batches, prompts_by_scene=None: {"tracks": []},
                    recover_with_text_prompt=lambda frame_batches, text_prompt: {"tracks": []},
                ),
                embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: []),
                label_client=SimpleNamespace(score_image_labels=lambda image_paths, labels: []),
            )
            planner_state, trace_path = run_agent_review_pass(
                inspect_state=inspect_state,
                tooling_runtime=tooling_runtime,
                bundle=bundle,
                max_actions=2,
            )
            self.assertLessEqual(len(planner_state.tool_calls), 2)
            self.assertTrue(Path(trace_path).exists())

    def test_agentic_tool_loop_rebuilds_missing_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame_path = root / "frame.jpg"
            Image.new("RGB", (64, 64), color="white").save(frame_path)

            inspect_state = {
                "video_path": str(root / "video.mp4"),
                "video_probe": SimpleNamespace(
                    duration_seconds=2.0,
                    fps=2.0,
                    width=64,
                    height=64,
                ),
                "candidate_cuts": [],
                "evidence_windows": [
                    EvidenceWindow(
                        window_id="window-0000",
                        start_time=0.0,
                        end_time=2.0,
                        sampled_frame_ids=[str(frame_path)],
                        artifact_refs=[str(root / "storyboard.jpg")],
                    )
                ],
                "frame_batches": [
                    FrameBatch(
                        scene_index=0,
                        frames=[
                            SampledFrame(
                                scene_index=0,
                            timestamp_seconds=0.0,
                            image_path=str(frame_path),
                        )
                    ],
                )
                ],
                "storyboard_path": str(root / "storyboard.jpg"),
                "sam3_track_set": Sam3TrackSet(
                    provider="fake-sam3",
                    strategy="prompt_free",
                    tracks=[
                        Sam3EntityTrack(
                            track_id="trk0",
                            scene_index=0,
                            start_seconds=0.0,
                            end_seconds=0.0,
                            confidence=0.9,
                            label_hint="object",
                            points=[
                                Sam3TrackPoint(
                                    timestamp_seconds=0.0,
                                    frame_path=str(frame_path),
                                    confidence=0.9,
                                    bbox_xyxy=[8.0, 8.0, 32.0, 32.0],
                                )
                            ],
                        )
                    ],
                ),
                "generation_groups": [],
                "sound_event_segments": [],
                "ambience_beds": [],
                "physical_sources": [],
            }

            updated_state, planner_state, trace_path = run_agentic_tool_loop(
                inspect_state=inspect_state,
                tooling_runtime=build_fake_tooling_runtime(),
                max_actions=1,
            )

            self.assertTrue(updated_state["track_crops"])
            self.assertIn("multitrack_bundle", updated_state)
            self.assertLessEqual(len(planner_state.tool_calls), 1)
            self.assertTrue(Path(trace_path).exists())

    def test_agentic_tool_loop_uses_final_writer_once_after_structural_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame_path = root / "frame.jpg"
            Image.new("RGB", (64, 64), color="white").save(frame_path)

            writer_calls: list[dict[str, object]] = []

            class _Writer:
                def write_group_description(self, context: dict[str, object]) -> object:
                    writer_calls.append(context)
                    return SimpleNamespace(
                        canonical_description="writer-final",
                        description_confidence=0.9,
                        description_rationale="writer-finalized",
                    )

            fake_runtime = build_fake_tooling_runtime()
            tooling_runtime = SimpleNamespace(
                sam3_client=fake_runtime.sam3_client,
                embedding_client=fake_runtime.embedding_client,
                label_client=fake_runtime.label_client,
                description_writer=_Writer(),
                adjudication_judge=None,
                should_release_clients=False,
                residency_mode="resident",
                resident_client_names=lambda: ["sam3", "embedding", "label", "description_writer"],
            )

            inspect_state = {
                "video_path": str(root / "video.mp4"),
                "video_probe": SimpleNamespace(
                    duration_seconds=2.0,
                    fps=2.0,
                    width=64,
                    height=64,
                ),
                "candidate_cuts": [],
                "evidence_windows": [
                    EvidenceWindow(
                        window_id="window-0000",
                        start_time=0.0,
                        end_time=2.0,
                        sampled_frame_ids=[str(frame_path)],
                        artifact_refs=[str(root / "storyboard.jpg")],
                    )
                ],
                "frame_batches": [
                    FrameBatch(
                        scene_index=0,
                        frames=[
                            SampledFrame(
                                scene_index=0,
                                timestamp_seconds=0.0,
                                image_path=str(frame_path),
                            )
                        ],
                    )
                ],
                "storyboard_path": str(root / "storyboard.jpg"),
                "sam3_track_set": Sam3TrackSet(
                    provider="fake-sam3",
                    strategy="prompt_free",
                    tracks=[
                        Sam3EntityTrack(
                            track_id="trk0",
                            scene_index=0,
                            start_seconds=0.0,
                            end_seconds=0.0,
                            confidence=0.9,
                            label_hint="object",
                            points=[
                                Sam3TrackPoint(
                                    timestamp_seconds=0.0,
                                    frame_path=str(frame_path),
                                    confidence=0.9,
                                    bbox_xyxy=[8.0, 8.0, 32.0, 32.0],
                                )
                            ],
                        )
                    ],
                ),
                "generation_groups": [],
                "sound_event_segments": [],
                "ambience_beds": [],
                "physical_sources": [],
            }

            updated_state, _, _ = run_agentic_tool_loop(
                inspect_state=inspect_state,
                tooling_runtime=tooling_runtime,
                max_actions=1,
            )

            self.assertGreaterEqual(len(writer_calls), 1)
            self.assertEqual(
                updated_state["multitrack_bundle"].pipeline_metadata["interim_description_writer_call_count"],
                0,
            )
            self.assertGreaterEqual(
                updated_state["multitrack_bundle"].pipeline_metadata["interim_bundle_build_count"],
                1,
            )
            self.assertEqual(
                updated_state["multitrack_bundle"].pipeline_metadata["description_writer_call_count"],
                updated_state["multitrack_bundle"].pipeline_metadata["final_description_writer_call_count"],
            )

    def test_build_issues_emits_hypothesis_and_description_repair_signals(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=10.0, fps=2.0, width=320, height=240),
            candidate_cuts=[],
            evidence_windows=[
                EvidenceWindow(
                    window_id="window-0000",
                    start_time=0.0,
                    end_time=9.5,
                )
            ],
            generation_groups=[],
            validation=ValidationReport(status="pass_with_warnings", issues=[]),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        bundle.generation_groups.append(
            SimpleNamespace(
                group_id="gen-0000",
                description_stale=True,
            )
        )
        inspect_state = {
            "video_probe": SimpleNamespace(duration_seconds=10.0),
            "candidate_cuts": [],
            "frame_batches": [],
            "sam3_track_set": Sam3TrackSet(provider="fake", tracks=[]),
            "track_label_candidates": {},
            "storyboard_path": "/tmp/run/storyboard.jpg",
            "verified_hypotheses_by_window": {
                0: {"uncertain_hypotheses": ["sword", "fighter"]}
            },
            "scene_hypotheses_by_window": {
                0: {"foreground_entities": ["fighter"], "candidate_sound_sources": ["sword"]}
            },
            "proposal_provenance_by_window": {0: {"ontology_extraction": ["fighter"]}},
        }

        issues = _build_issues(bundle=bundle, inspect_state=inspect_state)
        issue_types = {issue.issue_type for issue in issues}
        self.assertIn("hypothesis_conflict", issue_types)
        self.assertIn("description_stale", issue_types)

    def test_build_issues_promotes_missing_sources_into_foreground_repair(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=10.0, fps=2.0, width=320, height=240),
            candidate_cuts=[],
            evidence_windows=[
                EvidenceWindow(
                    window_id="window-0000",
                    start_time=0.0,
                    end_time=4.0,
                )
            ],
            generation_groups=[],
            validation=ValidationReport(
                status="pass_with_warnings",
                issues=[
                    ValidationIssue(
                        issue_type="missing_dominant_source",
                        severity="warning",
                        message="Bundle has no physical sources.",
                        repair_tool="extract_entities",
                    )
                ],
            ),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        inspect_state = {
            "video_path": "/tmp/video.mp4",
            "frame_batches": [],
            "sam3_track_set": Sam3TrackSet(provider="fake", tracks=[]),
        }

        issues = _build_issues(bundle=bundle, inspect_state=inspect_state)
        issue_types = {issue.issue_type for issue in issues}
        self.assertIn("foreground_collapse", issue_types)
        self.assertNotIn("validation_issue", issue_types)

    def test_grouping_ambiguity_payload_matches_group_acoustic_recipes(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=10.0, fps=2.0, width=320, height=240),
            evidence_windows=[
                EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=4.0)
            ],
            physical_sources=[],
            sound_events=[],
            ambience_beds=[],
            validation=ValidationReport(
                status="pass_with_warnings",
                issues=[
                    ValidationIssue(
                        issue_type="suspicious_cross_scene_generation_merge",
                        severity="warning",
                        message="merge looks suspicious",
                    )
                ],
            ),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        inspect_state = {
            "candidate_groups": ["cg0"],
            "track_routing_decisions": {"trk0": "tta"},
            "scene_hypotheses_by_window": {0: {"background_environment": ["street"]}},
            "proposal_provenance_by_window": {0: {"ontology_semantics": ["street"]}},
            "sam3_track_set": Sam3TrackSet(provider="fake", tracks=[]),
        }
        issues = _build_issues(bundle=bundle, inspect_state=inspect_state)
        grouping_issue = next(issue for issue in issues if issue.issue_type == "grouping_ambiguity")
        self.assertIn("sound_events", grouping_issue.payload)
        self.assertIn("ambience_beds", grouping_issue.payload)
        self.assertIn("physical_sources", grouping_issue.payload)
        self.assertIn("routing_decisions_by_track", grouping_issue.payload)
        self.assertIn("scene_hypotheses_by_window", grouping_issue.payload)
        self.assertIn("proposal_provenance_by_window", grouping_issue.payload)

    def test_build_issues_skips_low_confidence_identity_merge_when_structure_is_already_rich(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=10.0, fps=2.0, width=320, height=240),
            evidence_windows=[
                EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=4.0)
            ],
            physical_sources=[
                PhysicalSourceTrack(
                    source_id=f"source-{idx}",
                    kind="foreground",
                    label_candidates=[LabelCandidate(label="person", score=0.9)],
                    spans=[(0.0, 1.0)],
                    track_refs=[],
                    identity_confidence=0.9,
                    reid_neighbors=[],
                )
                for idx in range(6)
            ],
            sound_events=[
                SoundEventSegment(
                    event_id=f"event-{idx}",
                    source_id=f"source-{idx % 6}",
                    start_time=0.0,
                    end_time=1.0,
                    event_type="presence_texture",
                    confidence=0.8,
                )
                for idx in range(12)
            ],
            validation=ValidationReport(
                status="pass_with_warnings",
                issues=[
                    ValidationIssue(
                        issue_type="low_confidence_identity_merge",
                        severity="warning",
                        message="merge confidence is low",
                        related_ids=["source-0"],
                    )
                ],
            ),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        inspect_state = {
            "frame_batches": [],
            "sam3_track_set": Sam3TrackSet(provider="fake", tracks=[]),
        }
        issue_types = {issue.issue_type for issue in _build_issues(bundle=bundle, inspect_state=inspect_state)}
        self.assertNotIn("ambiguous_source", issue_types)

    def test_build_issues_skips_missing_crops_when_no_tracks_exist(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=8.0, fps=2.0, width=320, height=240),
            evidence_windows=[
                EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=4.0)
            ],
            validation=ValidationReport(status="pass_with_warnings", issues=[]),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        inspect_state = {
            "video_path": "/tmp/video.mp4",
            "frame_batches": [],
            "sam3_track_set": Sam3TrackSet(provider="fake", tracks=[]),
            "track_crops": [],
        }

        issue_types = {issue.issue_type for issue in _build_issues(bundle=bundle, inspect_state=inspect_state)}
        self.assertIn("foreground_collapse", issue_types)
        self.assertNotIn("missing_crops", issue_types)
        self.assertNotIn("missing_labels", issue_types)

    def test_adjudicator_receives_recovery_history_for_foreground_collapse(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=8.0, fps=2.0, width=320, height=240),
            evidence_windows=[
                EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=4.0)
            ],
            validation=ValidationReport(status="pass_with_warnings", issues=[]),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        issue = AgentIssue(
            issue_id="foreground-collapse",
            issue_type="foreground_collapse",
            description="No foreground tracks were extracted from a non-trivial clip.",
            payload={"frames_per_scene": 6},
        )
        captured: dict[str, object] = {}

        class _Judge:
            def judge_issue(self, context: dict[str, object]) -> object:
                captured.update(context)
                return SimpleNamespace(model_dump=lambda mode="json": {"resolution": "run_tool", "tool_name": "recover_foreground_sources", "rationale": "try prompts", "confidence": 0.8})

        tooling_runtime = SimpleNamespace(adjudication_judge=_Judge())
        inspect_state = {
            "frames_per_window": 6,
            "sam3_track_set": Sam3TrackSet(provider="fake", tracks=[]),
            "scene_prompt_candidates": {0: ["cat", "animal"]},
            "recovery_actions": ["densify_window_sampling"],
            "recovery_attempts": [{"tool_name": "densify_window_sampling", "frames_per_scene": 6}],
            "stage_history": [{"kind": "stage", "stage": "extract_entities", "track_count": 0}],
        }

        decision = _adjudicate_issue(
            inspect_state=inspect_state,
            bundle=bundle,
            issue=issue,
            planned_tool_name="recover_foreground_sources",
            tooling_runtime=tooling_runtime,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(captured["current_frames_per_window"], 6)
        self.assertEqual(captured["recovery_actions"], ["densify_window_sampling"])
        self.assertEqual(captured["recovery_attempts"], [{"tool_name": "densify_window_sampling", "frames_per_scene": 6}])
        self.assertEqual(captured["stage_history"], [{"kind": "stage", "stage": "extract_entities", "track_count": 0}])

    def test_adjudicator_failure_falls_back_to_deterministic_plan(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=8.0, fps=2.0, width=320, height=240),
            evidence_windows=[
                EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=4.0)
            ],
            validation=ValidationReport(status="pass_with_warnings", issues=[]),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        issue = AgentIssue(
            issue_id="foreground-collapse",
            issue_type="foreground_collapse",
            description="No foreground tracks were extracted from a non-trivial clip.",
            payload={},
        )

        class _BrokenJudge:
            def judge_issue(self, context: dict[str, object]) -> object:
                raise RuntimeError(f"quota exhausted for {context['issue_id']}")

        decision = _adjudicate_issue(
            inspect_state={
                "video_path": "/tmp/video.mp4",
                "frame_batches": [],
                "sam3_track_set": Sam3TrackSet(provider="fake", tracks=[]),
            },
            bundle=bundle,
            issue=issue,
            planned_tool_name="densify_window_sampling",
            tooling_runtime=SimpleNamespace(adjudication_judge=_BrokenJudge()),
        )
        self.assertIsNone(decision)
