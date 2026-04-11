from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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


def build_final_bundle(
    state: InspectState,
    *,
    description_writer: object | None = None,
) -> MultitrackDescriptionBundle:
    probe = state["video_probe"]
    generation_groups = synthesize_canonical_descriptions(
        list(state.get("generation_groups", [])),
        sound_events=list(state.get("sound_event_segments", [])),
        ambience_beds=list(state.get("ambience_beds", [])),
        physical_sources=list(state.get("physical_sources", [])),
        description_writer=description_writer,
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
            run_dir=_state_path(state.get("artifact_run_dir")),
            storyboard_path=_state_path(state.get("storyboard_path")),
            crop_dir=_artifact_dir_from_refs(state.get("track_crops", [])),
            clip_dir=_artifact_dir_from_refs(
                state.get("evidence_windows", []), attr="artifact_refs"
            ),
            trace_path=_state_path(state.get("agent_trace_path")),
            bundle_path=_state_path(state.get("bundle_path")),
            review_bundle_path=_state_path(state.get("review_bundle_path")),
        ),
        review_metadata=ReviewMetadata(),
        pipeline_metadata={
            "pipeline_version": getattr(
                state.get("options"), "pipeline_mode", "tool_first_foundation"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "tool_context_enabled": True,
            "sampling_frames_per_window": state.get("frames_per_window"),
            "recovery_actions": list(state.get("recovery_actions", [])),
            "recovery_attempts": list(state.get("recovery_attempts", [])),
            "stage_history": list(state.get("stage_history", [])),
            "runtime_trace_path": _state_path(state.get("runtime_trace_path")),
            "terminal_resolution": state.get("terminal_resolution"),
            "agent_review_decisions": list(state.get("agent_review_decisions", [])),
            "effective_runtime_profile": state.get("effective_runtime_profile"),
            "runtime_profile_source": state.get("runtime_profile_source"),
            "runtime_residency_mode": state.get("runtime_residency_mode"),
            "warm_start": state.get("warm_start"),
            "resident_models_before_run": list(
                state.get("resident_models_before_run", [])
            ),
        },
    )
    issues = validate_bundle(bundle)
    issues = _normalize_recovery_validation_issues(bundle, state=state, issues=issues)
    bundle.validation = ValidationReport(
        status="fail" if any(issue.severity == "error" for issue in issues) else ("pass_with_warnings" if issues else "pass"),
        issues=issues,
    )
    return bundle


def finalize_route_decision(group: object) -> RoutingDecision:
    existing = getattr(group, "route_decision", None)
    routing_candidate = _best_routing_candidate(
        getattr(group, "routing_candidates", None)
    )
    if routing_candidate is not None:
        return RoutingDecision(
            model_type=routing_candidate.model_type,
            confidence=round(routing_candidate.confidence, 4),
            factors=["group_routing_priors", routing_candidate.aggregate_method],
            reasoning=routing_candidate.reasoning,
            rule_based=True,
        )
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


def _best_routing_candidate(candidates: object) -> object | None:
    from v2a_inspect.tools.types import GroupRoutingDecision

    if not isinstance(candidates, list):
        return None
    typed_candidates = []
    for candidate in candidates:
        if isinstance(candidate, GroupRoutingDecision):
            typed_candidates.append(candidate)
        elif isinstance(candidate, dict):
            typed_candidates.append(GroupRoutingDecision.model_validate(candidate))
    if not typed_candidates:
        return None
    return max(typed_candidates, key=lambda candidate: candidate.confidence)


def _artifact_dir_from_refs(items: list[object], *, attr: str = "crop_path") -> str | None:
    for item in items:
        if attr == "artifact_refs":
            refs = list(getattr(item, attr, []))
            for ref in refs:
                if isinstance(ref, str) and ref:
                    return str(Path(ref).parent)
            continue
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return str(Path(value).parent)
    return None


def _state_path(value: object) -> str | None:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and value:
        return value
    return None


def _video_id_from_path(video_path: str) -> str:
    return Path(video_path or "video").stem or "video"


def _normalize_recovery_validation_issues(
    bundle: MultitrackDescriptionBundle,
    *,
    state: InspectState,
    issues: list[ValidationIssue],
) -> list[ValidationIssue]:
    if bundle.physical_sources:
        return issues
    attempts = list(state.get("recovery_attempts", []))
    resolution = state.get("terminal_resolution")
    if not attempts and resolution is None:
        return issues
    if resolution is None:
        return issues

    if resolution == "accepted_ambience_only":
        if not _looks_like_accepted_ambience_only(bundle):
            return issues
        filtered = [issue for issue in issues if issue.issue_type != "missing_dominant_source"]
        filtered.append(
            ValidationIssue(
                issue_type="accepted_ambience_only",
                severity="info",
                message="Bounded foreground recovery still yielded an ambience-only bundle that looks structurally consistent.",
                recommended_action="none",
            )
        )
        return filtered

    if resolution == "recovery_exhausted":
        filtered = [issue for issue in issues if issue.issue_type != "missing_dominant_source"]
        filtered.append(
            ValidationIssue(
                issue_type="recovery_exhausted",
                severity="warning",
                message="Foreground recovery attempts were exhausted without producing physical sources.",
                recommended_action="human_review",
            )
        )
    return filtered


def _looks_like_accepted_ambience_only(bundle: MultitrackDescriptionBundle) -> bool:
    if bundle.physical_sources or bundle.sound_events or not bundle.ambience_beds:
        return False
    if any(group.member_event_ids for group in bundle.generation_groups):
        return False
    if not bundle.generation_groups:
        return False
    total_duration = float(bundle.video_meta.duration_seconds or 0.0)
    if total_duration <= 0.0:
        return False
    ambience_coverage = sum(
        max(bed.end_time - bed.start_time, 0.0)
        for bed in bundle.ambience_beds
    )
    return ambience_coverage >= total_duration * 0.85
