from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from v2a_inspect.contracts import MultitrackDescriptionBundle


class DatasetRecord(BaseModel):
    record_id: str
    video_ref: str
    bundle: MultitrackDescriptionBundle
    evidence_refs: dict[str, str | None] = Field(default_factory=dict)
    review_metadata: dict[str, object] = Field(default_factory=dict)
    pipeline_version: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    validation_status: Literal["pass", "pass_with_warnings", "fail"]
    crop_evidence_enabled: bool = True
