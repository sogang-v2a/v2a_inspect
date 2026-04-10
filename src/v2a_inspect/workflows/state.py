from __future__ import annotations

from typing import Any, Literal

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
from v2a_inspect.pipeline.response_models import (
    GroupedAnalysis,
    RawTrack,
    TrackGroup,
    VideoSceneAnalysis,
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
    """User-configurable options for the inspection workflow."""

    fps: float = Field(default=2.0, gt=0.0)
    pipeline_mode: Literal[
        "legacy_gemini",
        "tool_first_foundation",
        "agentic_tool_first",
    ] = "agentic_tool_first"
    scene_analysis_mode: Literal["default", "extended"] = "default"
    enable_vlm_verify: bool = True
    enable_model_select: bool = False
    gemini_model: str = DEFAULT_GEMINI_MODEL
    upload_timeout_seconds: int = Field(default=300, ge=1)
    text_timeout_ms: int = Field(default=120_000, ge=1)
    video_timeout_ms: int = Field(default=180_000, ge=1)
    max_retries: int = Field(default=3, ge=0)
    poll_interval_seconds: float = Field(default=2.0, gt=0.0)
    runtime_mode: Literal["nvidia_docker", "in_process"] = "nvidia_docker"
    runtime_profile: Literal["mig10_safe", "full_gpu", "cpu_dev"] = "mig10_safe"
    remote_gpu_target: str = "sogang_gpu"
    minimum_gpu_vram_gb: int = Field(default=10, ge=1, le=80)
    server_base_url: str | None = None
    remote_timeout_seconds: int = Field(default=120, ge=1)
    remote_gpu_preference: str | None = None
    remote_gpu_fallback: str | None = None
    remote_gpu_vram_preference_gb: int = Field(default=10, ge=1, le=80)
    remote_gpu_vram_cap_gb: int = Field(default=80, ge=1, le=80)


class InspectState(TypedDict, total=False):
    """Shared LangGraph state for the inspection workflow."""

    video_path: str
    options: InspectOptions
    gemini_file: Any
    scene_analysis: VideoSceneAnalysis
    raw_tracks: list[RawTrack]
    text_groups: list[TrackGroup]
    verified_groups: list[TrackGroup]
    final_groups: list[TrackGroup]
    grouped_analysis: GroupedAnalysis
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
    frame_batches: list[FrameBatch]
    frames_per_window: int
    stage_history: list[dict[str, object]]
    storyboard_path: str
    runtime_trace_path: str
    sam3_track_set: Sam3TrackSet
    scene_prompt_candidates: dict[int, list[str]]
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
    tool_scene_summary: str
    tool_grouping_hints: str
    tool_verify_hints: str
    tool_routing_hints: str
    recovery_actions: list[str]
    recovery_attempts: list[dict[str, object]]
