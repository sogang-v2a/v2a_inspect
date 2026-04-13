from __future__ import annotations

import unittest

from v2a_inspect.tools.types import FrameBatch, SampledFrame
from v2a_inspect_server.scene_hypotheses import (
    SceneHypothesis,
    WindowOntologyExpansion,
    verify_scene_hypotheses,
)
from v2a_inspect_server.source_ontology import EXTRACTION_ENTITY_TERMS, SEMANTIC_HINT_TERMS


class SceneHypothesisTests(unittest.TestCase):
    def test_verify_scene_hypotheses_prefers_multi_source_support(self) -> None:
        frame_batches = [
            FrameBatch(
                scene_index=0,
                frames=[
                    SampledFrame(
                        scene_index=0,
                        timestamp_seconds=0.0,
                        image_path="/tmp/frame.jpg",
                    )
                ],
            )
        ]
        verified = verify_scene_hypotheses(
            frame_batches=frame_batches,
            ontology_scores={
                0: {
                    "extraction_entities": [
                        {"label": "sword", "score": 0.19},
                        {"label": "fighter", "score": 0.18},
                        {"label": "cloud", "score": 0.01},
                    ]
                }
            },
            scene_hypotheses={
                0: SceneHypothesis(
                    foreground_entities=["fighter"],
                    candidate_sound_sources=["sword"],
                    interactions=["fighting"],
                    material_cues=["metal"],
                    confidence=0.9,
                )
            },
            moving_region_labels={0: ["sword"]},
            expanded_candidates={
                0: WindowOntologyExpansion(
                    extraction_prompts=["sword", "fighter", "cloud"],
                    semantic_hints=["fighting", "metal"],
                    provenance={},
                )
            },
        )
        scene = verified[0]
        self.assertIn("sword", scene.verified_extraction_prompts)
        self.assertIn("fighter", scene.verified_extraction_prompts)
        self.assertIn("cloud", scene.rejected_hypotheses)
        self.assertIn("fighting", scene.verified_semantic_hints)

    def test_ontology_is_large_and_deduped(self) -> None:
        self.assertGreaterEqual(len(EXTRACTION_ENTITY_TERMS), 140)
        self.assertGreaterEqual(len(SEMANTIC_HINT_TERMS), 80)
        self.assertEqual(len(EXTRACTION_ENTITY_TERMS), len(set(EXTRACTION_ENTITY_TERMS)))
        self.assertEqual(len(SEMANTIC_HINT_TERMS), len(set(SEMANTIC_HINT_TERMS)))


if __name__ == "__main__":
    unittest.main()
