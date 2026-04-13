from __future__ import annotations

import unittest
from types import SimpleNamespace

from v2a_inspect.tools.types import FrameBatch, SampledFrame, Sam3TrackSet
from v2a_inspect_server.tool_registry import build_tool_registry


class _CapturingSam3Client:
    def __init__(self) -> None:
        self.prompts_by_scene: dict[int, list[str]] | None = None

    def extract_entities(self, frame_batches, *, prompts_by_scene=None, score_threshold=0.35, **kwargs):
        del frame_batches, score_threshold, kwargs
        self.prompts_by_scene = prompts_by_scene
        return Sam3TrackSet(provider="fake", strategy="scene_prompt_seeded", tracks=[])


class _FakeLabelClient:
    def score_image_labels(self, *, image_paths: list[str], labels: list[str]):
        del image_paths
        return [SimpleNamespace(label=label, score=max(0.0, 1.0 - index * 0.1)) for index, label in enumerate(labels)]


class ToolRegistryTests(unittest.TestCase):
    def test_registry_exposes_bundle_first_tool_surface(self) -> None:
        runtime = SimpleNamespace(
            runtime_profile="cpu_dev",
            should_release_clients=False,
            sam3_client=_CapturingSam3Client(),
            embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: []),
            label_client=_FakeLabelClient(),
            source_proposer=None,
            proposal_grounder=None,
            source_semantics_interpreter=None,
            grouping_judge=None,
            routing_judge=None,
            description_writer=None,
        )
        registry = build_tool_registry(runtime)
        self.assertIn("propose_source_hypotheses", registry)
        self.assertIn("verify_scene_hypotheses", registry)
        self.assertIn("build_source_semantics", registry)
        self.assertNotIn("recover_foreground_sources", registry)
        self.assertNotIn("group_acoustic_recipes", registry)
        self.assertNotIn("routing_priors", registry)

    def test_extract_entities_requires_explicit_grounded_prompts(self) -> None:
        sam3_client = _CapturingSam3Client()
        runtime = SimpleNamespace(
            runtime_profile="cpu_dev",
            should_release_clients=False,
            sam3_client=sam3_client,
            embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: []),
            label_client=_FakeLabelClient(),
            source_proposer=None,
            proposal_grounder=None,
            source_semantics_interpreter=None,
            grouping_judge=None,
            routing_judge=None,
            description_writer=None,
        )
        registry = build_tool_registry(runtime)
        frame_batches = [
            FrameBatch(
                scene_index=0,
                frames=[SampledFrame(scene_index=0, timestamp_seconds=0.5, image_path="/tmp/frame.jpg")],
            )
        ]
        registry["extract_entities"].handler(frame_batches=frame_batches, prompts_by_scene={0: ["table tennis paddle"]})
        self.assertEqual(sam3_client.prompts_by_scene, {0: ["table tennis paddle"]})

    def test_verify_scene_hypotheses_returns_unresolved_when_grounder_is_absent(self) -> None:
        runtime = SimpleNamespace(
            runtime_profile="cpu_dev",
            should_release_clients=False,
            sam3_client=_CapturingSam3Client(),
            embedding_client=SimpleNamespace(embed_images=lambda track_image_paths: []),
            label_client=_FakeLabelClient(),
            source_proposer=None,
            proposal_grounder=None,
            source_semantics_interpreter=None,
            grouping_judge=None,
            routing_judge=None,
            description_writer=None,
        )
        registry = build_tool_registry(runtime)
        frame_batches = [
            FrameBatch(
                scene_index=0,
                frames=[SampledFrame(scene_index=0, timestamp_seconds=0.5, image_path="/tmp/frame.jpg")],
            )
        ]
        verified = registry["verify_scene_hypotheses"].handler(
            frame_batches=frame_batches,
            scene_hypotheses_by_window={
                0: {
                    "visible_sources": ["paddle", "ball"],
                    "background_sources": [],
                    "interactions": ["paddle hits ball"],
                    "materials_surfaces": [],
                    "uncertain_regions": [],
                    "salient_regions": [],
                    "supporting_frame_indices": [0],
                    "confidence": 0.8,
                    "rationale": "visible sports objects",
                }
            },
            moving_regions_by_window={0: []},
            storyboard_path=None,
        )
        self.assertEqual(verified["prompts_by_scene"], {0: []})
        self.assertEqual(
            verified["verified_hypotheses_by_window"][0]["unresolved_phrases"],
            ["paddle", "ball", "paddle hits ball"],
        )


if __name__ == "__main__":
    unittest.main()
