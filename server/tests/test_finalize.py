from __future__ import annotations

import unittest
from types import SimpleNamespace

from v2a_inspect.contracts import (
    GenerationGroup,
    RoutingDecision,
)
from v2a_inspect.workflows import InspectOptions
from v2a_inspect_server.finalize import build_final_bundle


class FinalizeTests(unittest.TestCase):
    def test_build_final_bundle_preserves_unresolved_route_and_description(self) -> None:
        state = {
            "video_path": "/tmp/demo.mp4",
            "video_probe": SimpleNamespace(duration_seconds=5.0, fps=2.0, width=640, height=360),
            "candidate_cuts": [],
            "evidence_windows": [],
            "physical_sources": [],
            "sound_event_segments": [],
            "ambience_beds": [],
            "generation_groups": [
                GenerationGroup(
                    group_id="group-0000",
                    member_event_ids=["event-0000"],
                    canonical_label="paddle",
                    canonical_description=None,
                    description_origin=None,
                    group_confidence=0.8,
                    route_decision=None,
                )
            ],
            "options": InspectOptions(pipeline_mode="tool_first_foundation"),
        }
        bundle = build_final_bundle(state)
        self.assertEqual(bundle.validation.status, "pass_with_warnings")
        self.assertIsNone(bundle.generation_groups[0].route_decision)
        self.assertIsNone(bundle.generation_groups[0].canonical_description)
        self.assertEqual(
            {issue.issue_type for issue in bundle.validation.issues},
            {"unresolved_route_decision", "unresolved_description"},
        )

    def test_build_final_bundle_uses_writer_when_available(self) -> None:
        class _Writer:
            def write_group_description(self, context: dict[str, object]):
                del context
                return SimpleNamespace(
                    canonical_description="sharp paddle hit with short ball contact",
                    description_confidence=0.82,
                    description_rationale="writer-generated",
                )

        state = {
            "video_path": "/tmp/demo.mp4",
            "video_probe": SimpleNamespace(duration_seconds=5.0, fps=2.0, width=640, height=360),
            "candidate_cuts": [],
            "evidence_windows": [],
            "physical_sources": [],
            "sound_event_segments": [],
            "ambience_beds": [],
            "generation_groups": [
                GenerationGroup(
                    group_id="group-0000",
                    member_event_ids=["event-0000"],
                    canonical_label="paddle",
                    canonical_description=None,
                    description_origin=None,
                    group_confidence=0.8,
                    route_decision=RoutingDecision(
                        model_type="VTA",
                        confidence=0.7,
                        factors=[],
                        reasoning="visually synchronized strike",
                        decision_origin="gemini",
                    ),
                )
            ],
            "options": InspectOptions(pipeline_mode="tool_first_foundation"),
        }
        bundle = build_final_bundle(state, description_writer=_Writer())
        self.assertEqual(bundle.generation_groups[0].description_origin, "writer")
        self.assertEqual(
            bundle.generation_groups[0].canonical_description,
            "sharp paddle hit with short ball contact",
        )


if __name__ == "__main__":
    unittest.main()
