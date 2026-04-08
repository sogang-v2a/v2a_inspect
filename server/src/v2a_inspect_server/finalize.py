from __future__ import annotations

from datetime import UTC, datetime

from v2a_inspect.contracts import (
    ArtifactRefs,
    MultitrackDescriptionBundle,
    ReviewMetadata,
    RoutingDecision,
    ValidationIssue,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.workflows import InspectState

from .descriptions import synthesize_canonical_descriptions
from .validators import validate_bundle


def build_final_bundle(state: InspectState) -> MultitrackDescriptionBundle:
    probe = state["video_probe"]
    generation_groups = synthesize_canonical_descriptions(
        list(state.get("generation_groups", [])),
        sound_events=list(state.get("sound_event_segments", [])),
        ambience_beds=list(state.get("ambience_beds", [])),
        physical_sources=list(state.get("physical_sources", [])),
    )
    finalized_groups = [
        group.model_copy(update={"route_decision": finalize_route_decision(group)})
        for group in generation_groups
    ]

    bundle = MultitrackDescriptionBundle(
        video_id=_video_id_from_path(state.get("video_path", "video")),
        video_meta=VideoMeta(
            duration_seconds=probe.duration_seconds,
            fps=probe.fps,
            width=probe.width,
            height=probe.height,
        ),
        candidate_cuts=list(state.get("candidate_cuts", [])),
        evidence_windows=list(state.get("evidence_windows", [])),
        physical_sources=list(state.get("physical_sources", [])),
        sound_events=list(state.get("sound_event_segments", [])),
        ambience_beds=list(state.get("ambience_beds", [])),
        generation_groups=finalized_groups,
        validation=ValidationReport(status="pass_with_warnings"),
        artifacts=ArtifactRefs(
            storyboard_dir=state.get("storyboard_path"),
            crop_dir=_artifact_dir_from_refs(state.get("track_crops", [])),
            clip_dir=_artifact_dir_from_refs(state.get("evidence_windows", []), attr="artifact_refs"),
        ),
        review_metadata=ReviewMetadata(),
        pipeline_metadata={
            "pipeline_version": "stage6-foundation",
            "generated_at": datetime.now(UTC).isoformat(),
            "tool_context_enabled": True,
        },
    )
    issues = validate_bundle(bundle)
    bundle.validation = ValidationReport(
        status="fail" if any(issue.severity == "error" for issue in issues) else ("pass_with_warnings" if issues else "pass"),
        issues=issues,
    )
    return bundle


def finalize_route_decision(group: object) -> RoutingDecision:
    existing = getattr(group, "route_decision", None)
    member_event_ids = list(getattr(group, "member_event_ids", []))
    member_ambience_ids = list(getattr(group, "member_ambience_ids", []))
    if member_ambience_ids:
        return RoutingDecision(
            model_type="TTA",
            confidence=0.9,
            factors=["ambience_bed", "deterministic_prior"],
            reasoning="background ambience defaults to TTA in Stage 6 final routing",
            rule_based=True,
        )
    if existing is not None:
        model_type = getattr(existing, "model_type", "TTA")
        confidence = float(getattr(existing, "confidence", 0.5))
        factors = list(getattr(existing, "factors", [])) or ["existing_route"]
    else:
        model_type = "VTA" if len(member_event_ids) <= 1 else "TTA"
        confidence = 0.6 if len(member_event_ids) <= 1 else 0.7
        factors = ["event_cardinality"]
    if len(member_event_ids) > 3:
        model_type = "TTA"
        confidence = max(confidence, 0.75)
        factors = [*factors, "crowded_group"]
    elif len(member_event_ids) == 1:
        model_type = "VTA"
        confidence = max(confidence, 0.75)
        factors = [*factors, "single_dominant_event"]
    return RoutingDecision(
        model_type=model_type,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        factors=factors,
        reasoning="deterministic prior with final adjudicated normalization",
        rule_based=True,
    )


def _artifact_dir_from_refs(items: list[object], *, attr: str = "crop_path") -> str | None:
    for item in items:
        if attr == "artifact_refs":
            refs = list(getattr(item, attr, []))
            for ref in refs:
                if isinstance(ref, str) and ref:
                    return str(__import__("pathlib").Path(ref).parent)
            continue
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return str(__import__("pathlib").Path(value).parent)
    return None


def _video_id_from_path(video_path: str) -> str:
    from pathlib import Path

    return Path(video_path or "video").stem or "video"
