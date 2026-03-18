from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

NonNegativeInt = Annotated[int, Field(ge=0)]
ScoreValue = Annotated[float, Field(ge=1.0, le=5.0)]


class GroupingResponseGroup(BaseModel):
    member_indices: list[NonNegativeInt] = Field(default_factory=list)
    canonical_index: NonNegativeInt | None = None
    reasoning: str = ""


class GroupingResponse(BaseModel):
    groups: list[GroupingResponseGroup] = Field(default_factory=list)


class VLMVerifyResponse(BaseModel):
    same_entity: bool | Literal["uncertain"] = True
    confirmed_groups: list[list[NonNegativeInt]] | None = None
    reasoning: str = ""


class ModelSelectSegmentResponse(BaseModel):
    segment_index: NonNegativeInt | None = None
    motion_level: ScoreValue = 3.0
    event_coupling: ScoreValue = 3.0
    source_diversity: ScoreValue = 3.0
    reasoning: str = ""


class ModelSelectResponse(BaseModel):
    segments: list[ModelSelectSegmentResponse] = Field(default_factory=list)
