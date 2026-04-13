from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from v2a_inspect.contracts import (
    ArtifactRefs,
    MultitrackDescriptionBundle,
    ReviewMetadata,
    ValidationIssue,
    ValidationReport,
    VideoMeta,
)
from v2a_inspect.workflows import InspectState

from .descriptions import synthesize_canonical_descriptions
from .telemetry import record_stage, stage_start
from .validators import validate_bundle


def build_final_bundle(
    state: InspectState,
    *,
    description_writer: object | None = None,
) -> MultitrackDescriptionBundle:
    return _build_bundle(
        state,
        description_writer=description_writer,
        description_mode="final",
    )


def build_interim_bundle(
    state: InspectState,
) -> MultitrackDescriptionBundle:
    return _build_bundle(
        state,
        description_writer=None,
        description_mode="interim",
    )


def _build_bundle(
    state: InspectState,
    *,
    description_writer: object | None,
    description_mode: str,
) -> MultitrackDescriptionBundle:
    probe = state["video_probe"]
    description_started = stage_start()
    description_stats: dict[str, int] = {}
    generation_groups = synthesize_canonical_descriptions(
        list(state.get("generation_groups", [])),
        sound_events=list(state.get("sound_event_segments", [])),
        ambience_beds=list(state.get("ambience_beds", [])),
        physical_sources=list(state.get("physical_sources", [])),
        description_writer=description_writer,
        preserve_existing_descriptions=description_mode == "interim",
        stats=description_stats,
    )
    state[f"{description_mode}_bundle_build_count"] = (
        int(state.get(f"{description_mode}_bundle_build_count", 0)) + 1
    )
    state[f"{description_mode}_description_writer_call_count"] = (
        int(state.get(f"{description_mode}_description_writer_call_count", 0))
        + int(description_stats.get("description_writer_calls", 0))
    )
    state["description_writer_call_count"] = (
        int(state.get("description_writer_call_count", 0))
        + int(description_stats.get("description_writer_calls", 0))
    )
    record_stage(
        state,
        stage=f"{description_mode}_description_synthesis",
        started_at=description_started,
        metrics={
            "generation_group_count": len(generation_groups),
            "description_writer_calls": int(
                description_stats.get("description_writer_calls", 0)
            ),
            "preserved_description_count": int(
                description_stats.get("preserved_description_count", 0)
            ),
        },
    )
    bundle_started = stage_start()
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
        generation_groups=generation_groups,
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
            "analysis_media_mode": "silent_video",
            "analysis_video_path": _state_path(state.get("analysis_video_path")),
            "sampling_frames_per_window": state.get("frames_per_window"),
            "recovery_actions": list(state.get("recovery_actions", [])),
            "recovery_attempts": list(state.get("recovery_attempts", [])),
            "stage_history": list(state.get("stage_history", [])),
            "runtime_trace_path": _state_path(state.get("runtime_trace_path")),
            "scene_hypotheses_by_window": dict(
                state.get("scene_hypotheses_by_window", {})
            ),
            "verified_hypotheses_by_window": dict(
                state.get("verified_hypotheses_by_window", {})
            ),
            "proposal_provenance_by_window": dict(
                state.get("proposal_provenance_by_window", {})
            ),
            "recipe_signatures_by_group": dict(
                state.get("recipe_signatures_by_group", {})
            ),
            "terminal_resolution": state.get("terminal_resolution"),
            "agent_review_decisions": list(state.get("agent_review_decisions", [])),
            "effective_runtime_profile": state.get("effective_runtime_profile"),
            "runtime_profile_source": state.get("runtime_profile_source"),
            "runtime_residency_mode": state.get("runtime_residency_mode"),
            "warm_start": state.get("warm_start"),
            "resident_models_before_run": list(
                state.get("resident_models_before_run", [])
            ),
            "description_writer_call_count": int(
                state.get("description_writer_call_count", 0)
            ),
            "final_description_writer_call_count": int(
                state.get("final_description_writer_call_count", 0)
            ),
            "interim_description_writer_call_count": int(
                state.get("interim_description_writer_call_count", 0)
            ),
            "adjudicator_call_count": int(state.get("adjudicator_call_count", 0)),
            "final_bundle_build_count": int(state.get("final_bundle_build_count", 0)),
            "interim_bundle_build_count": int(
                state.get("interim_bundle_build_count", 0)
            ),
        },
    )
    issues = validate_bundle(bundle)
    issues = _normalize_recovery_validation_issues(bundle, state=state, issues=issues)
    bundle.validation = ValidationReport(
        status="fail" if any(issue.severity == "error" for issue in issues) else ("pass_with_warnings" if issues else "pass"),
        issues=issues,
    )
    record_stage(
        state,
        stage=f"build_{description_mode}_bundle",
        started_at=bundle_started,
        metrics={
            "validation_status": bundle.validation.status,
            "validation_issue_count": len(bundle.validation.issues),
            "physical_source_count": len(bundle.physical_sources),
            "sound_event_count": len(bundle.sound_events),
            "generation_group_count": len(bundle.generation_groups),
        },
    )
    return bundle


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
