from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.contracts import AmbienceBed, GenerationGroup, PhysicalSourceTrack, SoundEventSegment
from v2a_inspect.runtime import build_llm

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class MergeJudgment(BaseModel):
    should_merge: bool = False
    rationale: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GeminiGroupingJudge:
    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str,
        max_retries: int = 1,
        timeout_seconds: float = 45.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._llm: BaseChatModel | None = None

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(
                model=self._model,
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout_seconds=self._timeout_seconds,
            )
        return self._llm

    def judge_pair(self, *, left: Mapping[str, object], right: Mapping[str, object]) -> MergeJudgment | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = [
            SystemMessage(
                content=(
                    "Judge whether two video-only sound-event groups should share one canonical generation group. "
                    "Use only the structured evidence. Prefer not merging unless the visible source/mechanism looks semantically aligned."
                )
            ),
            HumanMessage(
                content=(
                    "Return a typed merge judgment.\n\n"
                    + json.dumps({"left": dict(left), "right": dict(right)}, indent=2, ensure_ascii=False)
                )
            ),
        ]
        structured_llm = self.llm.with_structured_output(MergeJudgment, method="json_schema")
        try:
            result = structured_llm.invoke(prompt)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(result, MergeJudgment):
            return result
        return MergeJudgment.model_validate(result)


def group_generation_groups(
    *,
    sound_events: Sequence[SoundEventSegment],
    ambience_beds: Sequence[AmbienceBed],
    physical_sources: Sequence[PhysicalSourceTrack],
    grouping_judge: GeminiGroupingJudge | None,
) -> list[GenerationGroup]:
    sources_by_id = {source.source_id: source for source in physical_sources}
    event_nodes = list(sound_events)
    parent = {event.event_id: event.event_id for event in event_nodes}
    merge_notes: dict[str, list[str]] = defaultdict(list)

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left_id: str, right_id: str, note: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            merge_notes[left_root].append(note)
            return
        parent[right_root] = left_root
        merge_notes[left_root].extend(merge_notes.pop(right_root, []))
        merge_notes[left_root].append(note)

    for left_index, left_event in enumerate(event_nodes):
        for right_event in event_nodes[left_index + 1 :]:
            if not _candidate_pair(left_event, right_event, sources_by_id):
                continue
            judgment = (
                grouping_judge.judge_pair(
                    left=_event_context(left_event, sources_by_id),
                    right=_event_context(right_event, sources_by_id),
                )
                if grouping_judge is not None
                else None
            )
            if judgment is not None and judgment.should_merge:
                union(left_event.event_id, right_event.event_id, judgment.rationale)

    grouped_event_ids: dict[str, list[str]] = defaultdict(list)
    for event in event_nodes:
        grouped_event_ids[find(event.event_id)].append(event.event_id)

    groups: list[GenerationGroup] = []
    for group_index, member_event_ids in enumerate(grouped_event_ids.values()):
        member_sources = [
            sources_by_id[event.source_id]
            for event in event_nodes
            if event.event_id in member_event_ids and event.source_id in sources_by_id
        ]
        canonical_label = _canonical_label(member_sources)
        group_id = f"group-{group_index:04d}"
        groups.append(
            GenerationGroup(
                group_id=group_id,
                member_event_ids=sorted(member_event_ids),
                canonical_label=canonical_label,
                canonical_description=None,
                description_origin=None,
                group_confidence=round(_group_confidence(member_sources), 4),
                route_decision=None,
                reasoning_summary=" | ".join(merge_notes.get(find(member_event_ids[0]), [])),
            )
        )

    for ambience in ambience_beds:
        groups.append(
            GenerationGroup(
                group_id=f"group-ambience-{ambience.ambience_id}",
                member_ambience_ids=[ambience.ambience_id],
                canonical_label=ambience.environment_type or ambience.acoustic_profile,
                canonical_description=None,
                description_origin=None,
                group_confidence=round(ambience.confidence, 4),
                route_decision=None,
                reasoning_summary="single ambience group",
            )
        )
    return groups


def _candidate_pair(
    left_event: SoundEventSegment,
    right_event: SoundEventSegment,
    sources_by_id: Mapping[str, PhysicalSourceTrack],
) -> bool:
    if left_event.source_id == right_event.source_id:
        return True
    left_source = sources_by_id.get(left_event.source_id)
    right_source = sources_by_id.get(right_event.source_id)
    left_label = left_source.label_candidates[0].label if left_source and left_source.label_candidates else ""
    right_label = right_source.label_candidates[0].label if right_source and right_source.label_candidates else ""
    temporal_gap = max(right_event.start_time - left_event.end_time, left_event.start_time - right_event.end_time, 0.0)
    return bool(left_label and left_label == right_label and temporal_gap <= 2.0)


def _event_context(
    event: SoundEventSegment,
    sources_by_id: Mapping[str, PhysicalSourceTrack],
) -> dict[str, object]:
    source = sources_by_id.get(event.source_id)
    return {
        "event_id": event.event_id,
        "source_id": event.source_id,
        "start_time": event.start_time,
        "end_time": event.end_time,
        "event_type": event.event_type,
        "material_or_surface": event.material_or_surface,
        "intensity": event.intensity,
        "texture": event.texture,
        "pattern": event.pattern,
        "source_label": source.label_candidates[0].label if source and source.label_candidates else "",
        "source_kind": source.kind if source is not None else "unknown",
    }


def _canonical_label(member_sources: Sequence[PhysicalSourceTrack]) -> str:
    for source in member_sources:
        if source.label_candidates:
            return source.label_candidates[0].label
    return "unresolved"


def _group_confidence(member_sources: Sequence[PhysicalSourceTrack]) -> float:
    if not member_sources:
        return 0.0
    return sum(source.identity_confidence for source in member_sources) / len(member_sources)
