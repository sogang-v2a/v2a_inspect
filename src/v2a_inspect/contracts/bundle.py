from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CutReason(BaseModel):
    kind: Literal[
        "shot_boundary",
        "composition_change",
        "motion_regime_change",
        "source_lifecycle_change",
        "label_context_change",
        "interaction_onset",
        "fallback_window",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class CandidateCut(BaseModel):
    cut_id: str
    timestamp_seconds: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[CutReason] = Field(default_factory=list)


class EvidenceWindow(BaseModel):
    window_id: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    sampled_frame_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    cut_refs: list[str] = Field(default_factory=list)
    rationale: str = ""


class TrackCrop(BaseModel):
    crop_id: str
    track_id: str
    scene_index: int = Field(ge=0)
    frame_path: str
    crop_path: str
    timestamp_seconds: float = Field(ge=0.0)
    bbox_xyxy: list[float] = Field(default_factory=list, min_length=4, max_length=4)
    mask_rle: str | None = None


class LabelCandidate(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class IdentityEdge(BaseModel):
    edge_id: str
    source_track_id: str
    target_track_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    same_window: bool = False
    temporal_gap_seconds: float = Field(ge=0.0)
    label_compatibility: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    accepted: bool = False
    rationale: str = ""


class PhysicalSourceTrack(BaseModel):
    source_id: str
    kind: Literal["foreground", "background_region", "ambience_region", "unknown"]
    label_candidates: list[LabelCandidate] = Field(default_factory=list)
    spans: list[tuple[float, float]] = Field(default_factory=list)
    track_refs: list[str] = Field(default_factory=list)
    crop_refs: list[str] = Field(default_factory=list)
    window_refs: list[str] = Field(default_factory=list)
    identity_confidence: float = Field(ge=0.0, le=1.0)
    reid_neighbors: list[str] = Field(default_factory=list)
    temporary_adapter_from: str | None = None


class SoundEventSegment(BaseModel):
    event_id: str
    source_id: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    event_type: str
    sync_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    motion_profile: str | None = None
    interaction_flags: list[str] = Field(default_factory=list)
    material_or_surface: str | None = None
    intensity: str | None = None
    texture: str | None = None
    pattern: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class AmbienceBed(BaseModel):
    ambience_id: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    environment_type: str
    acoustic_profile: str
    foreground_exclusion_notes: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class RoutingDecision(BaseModel):
    model_type: Literal["TTA", "VTA"]
    confidence: float = Field(ge=0.0, le=1.0)
    factors: list[str] = Field(default_factory=list)
    reasoning: str
    rule_based: bool = False


class GenerationGroup(BaseModel):
    group_id: str
    member_event_ids: list[str] = Field(default_factory=list)
    member_ambience_ids: list[str] = Field(default_factory=list)
    canonical_label: str
    canonical_description: str
    description_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    description_rationale: str | None = None
    group_confidence: float = Field(ge=0.0, le=1.0)
    route_decision: RoutingDecision
    reasoning_summary: str = ""
    routing_candidates: list[dict[str, object]] = Field(default_factory=list)
    temporary_adapter_from: str | None = None


class ValidationIssue(BaseModel):
    issue_id: str | None = None
    issue_type: Literal[
        "low_confidence_identity_merge",
        "suspicious_cross_scene_generation_merge",
        "missing_dominant_source",
        "overlapping_contradictory_assignments",
        "route_inconsistency",
        "overly_vague_description",
        "unreviewed_low_confidence_output",
    ]
    severity: Literal["info", "warning", "error"]
    message: str
    related_ids: list[str] = Field(default_factory=list)
    recommended_action: Literal[
        "none",
        "rerun_tool",
        "human_review",
        "override_route",
        "split_group",
        "merge_groups",
        "rename_source",
        "approve",
    ] = "none"
    repair_tool: str | None = None


class ReviewEdit(BaseModel):
    edit_id: str
    action: Literal[
        "route_override",
        "split_generation_group",
        "merge_generation_groups",
        "rename_source",
        "approve_issue",
        "mark_missing_extraction",
    ]
    target_ids: list[str] = Field(default_factory=list)
    payload: dict[str, object] = Field(default_factory=dict)
    author: str | None = None
    rationale: str = ""
    created_at: str | None = None


class ReviewMetadata(BaseModel):
    approval_status: Literal["unreviewed", "approved", "approved_with_notes"] = (
        "unreviewed"
    )
    notes: list[str] = Field(default_factory=list)
    applied_edits: list[ReviewEdit] = Field(default_factory=list)


class ValidationReport(BaseModel):
    status: Literal["pass", "pass_with_warnings", "fail"]
    issues: list[ValidationIssue] = Field(default_factory=list)
    reviewed_issue_ids: list[str] = Field(default_factory=list)


class ArtifactRefs(BaseModel):
    run_dir: str | None = None
    storyboard_path: str | None = None
    crop_dir: str | None = None
    clip_dir: str | None = None
    trace_path: str | None = None
    bundle_path: str | None = None
    review_bundle_path: str | None = None


class VideoMeta(BaseModel):
    duration_seconds: float = Field(ge=0.0)
    fps: float | None = Field(default=None, gt=0.0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class MultitrackDescriptionBundle(BaseModel):
    video_id: str
    video_meta: VideoMeta
    candidate_cuts: list[CandidateCut] = Field(default_factory=list)
    evidence_windows: list[EvidenceWindow] = Field(default_factory=list)
    physical_sources: list[PhysicalSourceTrack] = Field(default_factory=list)
    sound_events: list[SoundEventSegment] = Field(default_factory=list)
    ambience_beds: list[AmbienceBed] = Field(default_factory=list)
    generation_groups: list[GenerationGroup] = Field(default_factory=list)
    validation: ValidationReport
    artifacts: ArtifactRefs = Field(default_factory=ArtifactRefs)
    review_metadata: ReviewMetadata = Field(default_factory=ReviewMetadata)
    pipeline_metadata: dict[str, object] = Field(default_factory=dict)
