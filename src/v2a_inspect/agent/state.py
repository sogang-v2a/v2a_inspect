from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class RetryBudget(BaseModel):
    extraction_retry_limit: int = Field(default=1, ge=0)
    manual_recovery_limit: int = Field(default=1, ge=0)
    foreground_recovery_limit: int = Field(default=3, ge=0)
    regroup_retry_limit: int = Field(default=2, ge=0)
    validation_round_limit: int = Field(default=3, ge=0)


class AgentIssue(BaseModel):
    issue_id: str
    issue_type: Literal[
        "structural_gap",
        "ambiguous_source",
        "foreground_collapse",
        "hypothesis_conflict",
        "missing_sources",
        "grouping_ambiguity",
        "routing_ambiguity",
        "description_stale",
    ]
    description: str
    priority: int = Field(default=100, ge=0)
    payload: dict[str, object] = Field(default_factory=dict)
    attempts: int = Field(default=0, ge=0)
    status: Literal["open", "resolved", "failed", "skipped"] = "open"


class PlannedAction(BaseModel):
    issue_id: str
    tool_name: str
    request_payload: dict[str, object] = Field(default_factory=dict)
    rationale: str


class ToolCallRecord(BaseModel):
    call_id: str
    issue_id: str
    tool_name: str
    request_payload: dict[str, object] = Field(default_factory=dict)
    effective_request_payload: dict[str, object] = Field(default_factory=dict)
    dropped_request_keys: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DecisionRecord(BaseModel):
    issue_id: str
    decision: Literal["accept", "reject", "retry", "skip"]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PlannerState(BaseModel):
    video_id: str
    issues: list[AgentIssue] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    retry_budget: RetryBudget = Field(default_factory=RetryBudget)
