from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from v2a_inspect.contracts import (
    ArtifactRefs,
    MultitrackDescriptionBundle,
    ValidationIssue,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.tools.types import FrameBatch, SampledFrame
from v2a_inspect_server.agentic import run_agent_review_pass


class AgenticIntegrationTests(unittest.TestCase):
    def test_agent_review_pass_logs_bounded_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storyboard_dir = Path(tmp_dir)
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
                artifacts=ArtifactRefs(storyboard_dir=str(storyboard_dir)),
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
