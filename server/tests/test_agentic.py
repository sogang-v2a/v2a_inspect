from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from v2a_inspect.agent import AgentIssue
from v2a_inspect.contracts import ArtifactRefs, EvidenceWindow, GenerationGroup, MultitrackDescriptionBundle, ValidationIssue, ValidationReport, VideoMeta
from v2a_inspect_server.agentic import _adjudicate_issue, _build_issues, run_agent_review_pass


class AgenticIntegrationTests(unittest.TestCase):
    def test_agent_review_pass_logs_bounded_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            bundle = MultitrackDescriptionBundle(
                video_id="video",
                video_meta=VideoMeta(duration_seconds=1.0, fps=2.0, width=320, height=240),
                evidence_windows=[EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=1.0)],
                generation_groups=[
                    GenerationGroup(
                        group_id="group-0000",
                        member_event_ids=["event-0000"],
                        canonical_label="paddle",
                        canonical_description="stale description",
                        description_origin="writer",
                        description_stale=True,
                        group_confidence=0.7,
                        route_decision=None,
                    )
                ],
                validation=ValidationReport(status="pass_with_warnings", issues=[]),
                artifacts=ArtifactRefs(run_dir=str(run_dir), storyboard_path=str(run_dir / "storyboard.jpg")),
            )
            inspect_state = {
                "video_path": "/tmp/video.mp4",
                "frame_batches": [],
                "sam3_track_set": SimpleNamespace(tracks=[]),
            }
            tooling_runtime = SimpleNamespace(
                runtime_profile="cpu_dev",
                should_release_clients=False,
                sam3_client=SimpleNamespace(extract_entities=lambda frame_batches, prompts_by_scene=None: {"tracks": []}),
                embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: []),
                label_client=SimpleNamespace(score_image_labels=lambda image_paths, labels: []),
                source_proposer=None,
                proposal_grounder=None,
                source_semantics_interpreter=None,
                grouping_judge=None,
                routing_judge=None,
                description_writer=None,
            )
            planner_state, trace_path = run_agent_review_pass(
                inspect_state=inspect_state,
                tooling_runtime=tooling_runtime,
                bundle=bundle,
                max_actions=1,
            )
            self.assertLessEqual(len(planner_state.tool_calls), 1)
            self.assertTrue(Path(trace_path).exists())

    def test_build_issues_emits_hypothesis_and_description_repair_signals(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=10.0, fps=2.0, width=320, height=240),
            candidate_cuts=[],
            evidence_windows=[EvidenceWindow(window_id="window-0000", start_time=0.0, end_time=9.5)],
            generation_groups=[
                GenerationGroup(
                    group_id="group-0000",
                    member_event_ids=["event-0000"],
                    canonical_label="paddle",
                    canonical_description="old",
                    description_origin="writer",
                    description_stale=True,
                    group_confidence=0.8,
                    route_decision=None,
                )
            ],
            validation=ValidationReport(status="pass_with_warnings", issues=[]),
            artifacts=ArtifactRefs(run_dir="/tmp/run", storyboard_path="/tmp/run/storyboard.jpg"),
        )
        inspect_state = {
            "video_probe": SimpleNamespace(duration_seconds=10.0),
            "candidate_cuts": [],
            "frame_batches": [],
            "sam3_track_set": SimpleNamespace(tracks=[]),
            "track_label_candidates": {},
            "storyboard_path": "/tmp/run/storyboard.jpg",
            "verified_hypotheses_by_window": {0: {"uncertain_hypotheses": ["sword", "fighter"]}},
            "scene_hypotheses_by_window": {0: {"foreground_entities": ["fighter"], "candidate_sound_sources": ["sword"]}},
            "proposal_provenance_by_window": {0: {"proposal_mode": "gemini_open_world"}},
        }
        issues = _build_issues(bundle=bundle, inspect_state=inspect_state)
        issue_types = {issue.issue_type for issue in issues}
        self.assertIn("hypothesis_conflict", issue_types)
        self.assertIn("description_stale", issue_types)

    def test_adjudicator_failure_falls_back_to_none(self) -> None:
        issue = AgentIssue(
            issue_id="issue-1",
            issue_type="description_stale",
            description="rewrite needed",
            payload={},
        )
        bundle = MultitrackDescriptionBundle(
            video_id="video",
            video_meta=VideoMeta(duration_seconds=1.0, fps=2.0, width=320, height=240),
            validation=ValidationReport(
                status="pass_with_warnings",
                issues=[ValidationIssue(issue_type="unresolved_description", severity="warning", message="missing")],
            ),
        )
        inspect_state = {"stage_history": [], "recovery_actions": [], "recovery_attempts": [], "frames_per_window": 2, "sam3_track_set": SimpleNamespace(tracks=[])}

        class _BrokenJudge:
            def judge_issue(self, context: dict[str, object]) -> object:
                del context
                raise RuntimeError("boom")

        tooling_runtime = SimpleNamespace(adjudication_judge=_BrokenJudge())
        decision = _adjudicate_issue(
            inspect_state=inspect_state,
            bundle=bundle,
            issue=issue,
            planned_tool_name="rerun_description_writer",
            tooling_runtime=tooling_runtime,
        )
        self.assertIsNone(decision)
        self.assertTrue(inspect_state["adjudicator_disabled_after_failure"])


if __name__ == "__main__":
    unittest.main()
