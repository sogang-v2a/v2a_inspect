from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2a_inspect.contracts import EvidenceWindow, GenerationGroup, RoutingDecision
from v2a_inspect.local_pipeline import run_local_inspect_raw
from v2a_inspect.local_runtime import RemoteRuntimeSnapshot
from v2a_inspect.tools.types import FrameBatch, SampledFrame, Sam3TrackSet, VideoProbe
from v2a_inspect.workflows import InspectOptions


class LocalPipelineTests(unittest.TestCase):
    def test_run_local_inspect_raw_executes_local_orchestration_and_persists_bundle_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            video_path = root / "clip.mp4"
            video_path.write_bytes(b"video")
            frame_path = root / "frame.jpg"
            frame_path.write_bytes(b"frame")
            storyboard_path = root / "storyboard.jpg"
            storyboard_path.write_bytes(b"storyboard")
            artifact_root = root / "artifacts"
            artifact_root.mkdir()

            evidence_windows = [
                EvidenceWindow(
                    window_id="window-0000",
                    start_time=0.0,
                    end_time=1.0,
                    sampled_frame_ids=[str(frame_path)],
                )
            ]
            frame_batches = [
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
            ]
            generation_group = GenerationGroup(
                group_id="group-0000",
                canonical_label="paddle",
                canonical_description="sharp paddle hit",
                description_origin="manual",
                description_stale=False,
                group_confidence=0.9,
                route_decision=RoutingDecision(
                    model_type="VTA",
                    confidence=0.8,
                    factors=["sync"],
                    reasoning="visual sync matters",
                    decision_origin="manual",
                ),
            )
            fake_runtime = SimpleNamespace(
                description_writer=None,
                residency_mode="remote_inference",
                remote_runtime_snapshot=lambda: RemoteRuntimeSnapshot(
                    effective_runtime_profile="full_gpu",
                    runtime_profile_source="remote_runtime_info",
                    residency_mode="resident",
                    resident_models=["sam3", "embedding", "label"],
                ),
            )
            registry = {
                "structural_overview": SimpleNamespace(
                    handler=lambda **kwargs: {
                        "probe": VideoProbe(video_path=str(video_path), duration_seconds=1.0, fps=2.0, width=320, height=240),
                        "candidate_cuts": [],
                        "evidence_windows": evidence_windows,
                        "frame_batches": frame_batches,
                        "storyboard_path": str(storyboard_path),
                        "artifact_root": str(artifact_root),
                        "frames_per_scene": 2,
                        "analysis_video_path": str(video_path),
                    }
                ),
                "propose_source_hypotheses": SimpleNamespace(
                    handler=lambda **kwargs: {
                        "scene_hypotheses_by_window": {0: {"visible_sources": ["paddle"], "source_cards": []}},
                        "moving_regions_by_window": {0: []},
                        "proposal_provenance_by_window": {0: {"source_card_count": 0}},
                        "warnings": [],
                    }
                ),
                "verify_scene_hypotheses": SimpleNamespace(
                    handler=lambda **kwargs: {
                        "verified_hypotheses_by_window": {0: {"extraction_prompts": ["paddle"], "semantic_hints": [], "unresolved_phrases": []}},
                        "region_seeds_by_scene": {0: []},
                        "prompts_by_scene": {0: ["paddle"]},
                        "proposal_provenance_by_window": {0: {"grounded_source_card_count": 0}},
                        "warnings": [],
                    }
                ),
                "extract_entities": SimpleNamespace(
                    handler=lambda **kwargs: Sam3TrackSet(provider="fake", strategy="scene_prompt_seeded", tracks=[])
                ),
                "crop_tracks": SimpleNamespace(handler=lambda **kwargs: []),
                "embed_track_crops": SimpleNamespace(handler=lambda **kwargs: []),
                "score_track_labels": SimpleNamespace(handler=lambda **kwargs: {}),
                "group_embeddings": SimpleNamespace(handler=lambda **kwargs: SimpleNamespace(groups=[])),
                "refine_candidate_cuts": SimpleNamespace(
                    handler=lambda **kwargs: {"candidate_cuts": [], "evidence_windows": evidence_windows}
                ),
                "build_source_semantics": SimpleNamespace(
                    handler=lambda **kwargs: {
                        "identity_edges": [],
                        "physical_sources": [],
                        "sound_events": [],
                        "ambience_beds": [],
                        "generation_groups": [generation_group],
                        "recipe_signatures": {},
                        "recipe_grouping_seconds": 0.0,
                    }
                ),
            }
            with patch("v2a_inspect.local_pipeline.LocalToolingRuntime", return_value=fake_runtime), patch(
                "v2a_inspect.local_pipeline.build_tool_registry", return_value=registry
            ):
                payload = run_local_inspect_raw(
                    video_path=str(video_path),
                    options=InspectOptions(server_base_url="http://server:8080", pipeline_mode="tool_first_foundation"),
                )
                bundle = payload["multitrack_bundle"]
                self.assertEqual(bundle["pipeline_metadata"]["effective_runtime_profile"], "full_gpu")
                self.assertEqual(bundle["pipeline_metadata"]["runtime_profile_source"], "remote_runtime_info")
                self.assertEqual(bundle["pipeline_metadata"]["resident_models_after_run"], ["sam3", "embedding", "label"])
                bundle_path = Path(bundle["artifacts"]["bundle_path"])
                persisted = json.loads(bundle_path.read_text(encoding="utf-8"))
                self.assertEqual(persisted["artifacts"]["bundle_path"], str(bundle_path))


if __name__ == "__main__":
    unittest.main()
