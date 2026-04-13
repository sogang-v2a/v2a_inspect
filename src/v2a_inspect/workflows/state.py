from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.contracts import (
    AmbienceBed,
    CandidateCut,
    EvidenceWindow,
    IdentityEdge,
    LabelCandidate,
    MultitrackDescriptionBundle,
    PhysicalSourceTrack,
    SoundEventSegment,
    TrackCrop,
)
from v2a_inspect.tools import (
    CandidateGroup,
    EntityEmbedding,
    FrameBatch,
    GroupRoutingDecision,
    Sam3TrackSet,
    SceneBoundary,
    TrackRoutingDecision,
    VideoProbe,
)


class InspectOptions(BaseModel):
    """User-configurable options for the supported server-backed inspection modes."""

    fps: float = Field(default=2.0, gt=0.0)
    pipeline_mode: Literal[
        "tool_first_foundation",
        "agentic_tool_first",
    ] = "agentic_tool_first"
    scene_analysis_mode: Literal["default", "extended"] = "default"
    gemini_model: str = DEFAULT_GEMINI_MODEL
    text_timeout_ms: int = Field(default=120_000, ge=1)
    max_retries: int = Field(default=3, ge=0)
    runtime_mode: Literal["nvidia_docker", "in_process"] = "nvidia_docker"
    runtime_profile: Literal["mig10_safe", "full_gpu", "cpu_dev"] = "full_gpu"
    remote_gpu_target: str = "sogang_gpu"
    minimum_gpu_vram_gb: int = Field(default=10, ge=1, le=80)
    server_base_url: str | None = None
    remote_timeout_seconds: int = Field(default=120, ge=1)
    remote_gpu_preference: str | None = None
    remote_gpu_fallback: str | None = None
    remote_gpu_vram_preference_gb: int = Field(default=10, ge=1, le=80)
    remote_gpu_vram_cap_gb: int = Field(default=80, ge=1, le=80)


class InspectState(TypedDict, total=False):
    video_path: str
    options: InspectOptions
    multitrack_bundle: MultitrackDescriptionBundle
    trace_id: str
    root_observation_id: str
    errors: list[str]
    warnings: list[str]
    progress_messages: list[str]
    video_probe: VideoProbe
    candidate_cuts: list[CandidateCut]
    evidence_windows: list[EvidenceWindow]
    scene_boundaries: list[SceneBoundary]
    analysis_video_path: str
    frame_batches: list[FrameBatch]
    frames_per_window: int
    stage_history: list[dict[str, object]]
    storyboard_path: str
    runtime_trace_path: str
    sam3_track_set: Sam3TrackSet
    scene_prompt_candidates: dict[int, list[str]]
    scene_hypotheses_by_window: dict[int, dict[str, object]]
    proposal_provenance_by_window: dict[int, dict[str, object]]
    verified_hypotheses_by_window: dict[int, dict[str, object]]
    track_crops: list[TrackCrop]
    entity_embeddings: list[EntityEmbedding]
    track_label_candidates: dict[str, list[LabelCandidate]]
    candidate_groups: list[CandidateGroup]
    routing_decisions: list[GroupRoutingDecision]
    track_routing_decisions: dict[str, TrackRoutingDecision]
    identity_edges: list[IdentityEdge]
    physical_sources: list[PhysicalSourceTrack]
    sound_event_segments: list[SoundEventSegment]
    ambience_beds: list[AmbienceBed]
    recipe_signatures_by_group: dict[str, dict[str, object]]
    recovery_actions: list[str]
    recovery_attempts: list[dict[str, object]]
    terminal_resolution: str
    agent_review_decisions: list[dict[str, object]]
    effective_runtime_profile: str
    runtime_profile_source: str
    runtime_residency_mode: str
    warm_start: bool
    resident_models_before_run: list[str]
