from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldSetClip(BaseModel):
    clip_id: str
    filename: str
    category: str
    visible_sources: list[str] = Field(default_factory=list)
    event_notes: list[str] = Field(default_factory=list)
    grouping_notes: list[str] = Field(default_factory=list)
    routing_notes: list[str] = Field(default_factory=list)


class GoldSetManifest(BaseModel):
    version: str
    clips: list[GoldSetClip] = Field(default_factory=list)


def load_gold_set_manifest(path: str | Path) -> GoldSetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldSetManifest.model_validate(payload)
