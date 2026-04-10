from __future__ import annotations

import unittest

from v2a_inspect.contracts import (
    GenerationGroup,
    MultitrackDescriptionBundle,
    PhysicalSourceTrack,
    RoutingDecision,
    SoundEventSegment,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.dataset import build_dataset_record
from v2a_inspect.evaluation import (
    build_downstream_generation_hooks,
    compare_baselines,
    crop_evidence_ablation,
    structural_metrics,
)


def _bundle(video_id: str, *, group_count: int, route_type: str, crop_evidence_enabled: bool = True) -> MultitrackDescriptionBundle:
    return MultitrackDescriptionBundle(
        video_id=video_id,
        video_meta=VideoMeta(duration_seconds=3.0, fps=2.0, width=320, height=240),
        physical_sources=[PhysicalSourceTrack(source_id="source-0000", kind="foreground", spans=[(0.0, 1.0)], track_refs=["trk0"], identity_confidence=0.9, reid_neighbors=[])],
        sound_events=[SoundEventSegment(event_id="event-0000", source_id="source-0000", start_time=0.0, end_time=1.0, event_type="continuous_motion", confidence=0.8)],
        generation_groups=[
            GenerationGroup(group_id=f"gen-{index:04d}", member_event_ids=["event-0000"] if index == 0 else [], canonical_label="label", canonical_description="desc", group_confidence=0.8, route_decision=RoutingDecision(model_type=route_type, confidence=0.8, factors=[], reasoning="", rule_based=True))
            for index in range(group_count)
        ],
        validation=ValidationReport(status="pass"),
        pipeline_metadata={"pipeline_version": "eval", "crop_evidence_enabled": crop_evidence_enabled},
    )


class EvaluationTests(unittest.TestCase):
    def test_structural_metrics_and_baselines(self) -> None:
        reference = build_dataset_record(video_ref="ref.mp4", bundle=_bundle("ref", group_count=1, route_type="TTA"))
        candidate = build_dataset_record(video_ref="cand.mp4", bundle=_bundle("cand", group_count=2, route_type="VTA"))
        metrics = structural_metrics(reference, candidate)
        self.assertIn("route_agreement", metrics)
        results = compare_baselines(reference=reference, candidates={"agentic": candidate})
        self.assertEqual(results[0].strategy, "agentic")

    def test_downstream_hooks_and_crop_ablation(self) -> None:
        with_crops = build_dataset_record(video_ref="with.mp4", bundle=_bundle("with", group_count=1, route_type="TTA"), crop_evidence_enabled=True)
        without_crops = build_dataset_record(video_ref="without.mp4", bundle=_bundle("without", group_count=2, route_type="TTA"), crop_evidence_enabled=False)
        hooks = build_downstream_generation_hooks(with_crops)
        self.assertEqual(len(hooks), 5)
        ablation = crop_evidence_ablation(with_crops=with_crops, without_crops=without_crops)
        self.assertIn("source_coverage_delta", ablation)
