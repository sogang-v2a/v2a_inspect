from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from v2a_inspect.contracts import (
    ArtifactRefs,
    EvidenceWindow,
    MultitrackDescriptionBundle,
    ValidationIssue,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.tools.types import FrameBatch, SampledFrame, Sam3EntityTrack, Sam3TrackPoint, Sam3TrackSet
from server.tests.fakes import build_fake_tooling_runtime
from v2a_inspect_server.agentic import run_agent_review_pass, run_agentic_tool_loop


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

    def test_agentic_tool_loop_rebuilds_missing_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame_path = root / "frame.jpg"
            Image.new("RGB", (64, 64), color="white").save(frame_path)

            inspect_state = {
                "video_path": str(root / "video.mp4"),
                "video_probe": SimpleNamespace(
                    duration_seconds=1.0,
                    fps=2.0,
                    width=64,
                    height=64,
                ),
                "candidate_cuts": [],
                "evidence_windows": [
                    EvidenceWindow(
                        window_id="window-0000",
                        start_time=0.0,
                        end_time=1.0,
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
