from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2a_inspect.contracts import (
    ArtifactRefs,
    GenerationGroup,
    MultitrackDescriptionBundle,
    PhysicalSourceTrack,
    RoutingDecision,
    SoundEventSegment,
    ValidationIssue,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.review import (
    apply_route_override,
    approve_validation_issue,
    load_bundle,
    merge_generation_groups,
    persist_bundle,
    rename_source,
    split_generation_group,
)


def _bundle() -> MultitrackDescriptionBundle:
    return MultitrackDescriptionBundle(
        video_id="vid-001",
        video_meta=VideoMeta(duration_seconds=3.0, fps=2.0, width=320, height=240),
        physical_sources=[
            PhysicalSourceTrack(
                source_id="source-0000",
                kind="foreground",
                spans=[(0.0, 1.0)],
                evidence_refs=["trk0"],
                identity_confidence=0.9,
                reid_neighbors=[],
            )
        ],
        sound_events=[
            SoundEventSegment(
                event_id="event-0000",
                source_id="source-0000",
                start_time=0.0,
                end_time=1.0,
                event_type="continuous_motion",
                confidence=0.9,
            ),
            SoundEventSegment(
                event_id="event-0001",
                source_id="source-0000",
                start_time=1.0,
                end_time=2.0,
                event_type="contact_event",
                confidence=0.8,
            ),
        ],
        generation_groups=[
            GenerationGroup(
                group_id="gen-0000",
                member_event_ids=["event-0000", "event-0001"],
                canonical_label="person:mix",
                canonical_description="placeholder",
                group_confidence=0.9,
                route_decision=RoutingDecision(
                    model_type="VTA",
                    confidence=0.7,
                    factors=[],
                    reasoning="",
                    rule_based=True,
                ),
            ),
            GenerationGroup(
                group_id="gen-0001",
                member_event_ids=[],
                member_ambience_ids=[],
                canonical_label="empty",
                canonical_description="empty",
                group_confidence=0.2,
                route_decision=RoutingDecision(
                    model_type="TTA",
                    confidence=0.5,
                    factors=[],
                    reasoning="",
                    rule_based=True,
                ),
            ),
        ],
        validation=ValidationReport(
            status="pass_with_warnings",
            issues=[
                ValidationIssue(
                    issue_id="issue-0000",
                    issue_type="overly_vague_description",
                    severity="warning",
                    message="needs review",
                )
            ],
        ),
        artifacts=ArtifactRefs(),
    )


class ReviewTests(unittest.TestCase):
    def test_route_override_and_approval_persist(self) -> None:
        bundle = approve_validation_issue(
            apply_route_override(
                _bundle(), group_id="gen-0000", model_type="TTA", author="tester"
            ),
            issue_id="issue-0000",
            author="tester",
        )
        self.assertEqual(bundle.generation_groups[0].route_decision.model_type, "TTA")
        self.assertIn("issue-0000", bundle.validation.reviewed_issue_ids)
        self.assertEqual(len(bundle.review_metadata.applied_edits), 2)

    def test_split_merge_rename_and_persist_bundle(self) -> None:
        bundle = rename_source(
            _bundle(), source_id="source-0000", new_label="runner", author="tester"
        )
        bundle = split_generation_group(
            bundle, group_id="gen-0000", event_ids=["event-0001"], author="tester"
        )
        bundle = merge_generation_groups(
            bundle, source_group_ids=["gen-0000", "gen-0000-split-01"], author="tester"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = persist_bundle(bundle, Path(tmp_dir) / "bundle.json")
            loaded = load_bundle(path)
        self.assertEqual(loaded.physical_sources[0].label_candidates[0].label, "runner")
        self.assertTrue(loaded.review_metadata.applied_edits)
