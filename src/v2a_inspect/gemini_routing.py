from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.contracts import AmbienceBed, GenerationGroup, PhysicalSourceTrack, RoutingDecision, SoundEventSegment
from v2a_inspect.runtime import build_llm, invoke_structured_llm

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class RoutingJudgment(BaseModel):
    model_type: str = Field(default="TTA")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


class GeminiRoutingJudge:
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

    def judge_group(self, *, context: Mapping[str, object]) -> RoutingJudgment | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = [
            SystemMessage(
                content=(
                    "Choose TTA or VTA for one generation group using only structured visual evidence. "
                    "Prefer TTA for coarse/ambient/texture-oriented groups and VTA when visual synchrony matters."
                )
            ),
            HumanMessage(content=json.dumps(dict(context), indent=2, ensure_ascii=False)),
        ]
        try:
            result = invoke_structured_llm(
                llm=self.llm,
                schema_model=RoutingJudgment,
                prompt=prompt,
                method="json_schema",
            )
        except Exception:  # noqa: BLE001
            return None
        if isinstance(result, RoutingJudgment):
            return result
        return RoutingJudgment.model_validate(result)


def route_generation_groups(
    *,
    generation_groups: Sequence[GenerationGroup],
    sound_events: Sequence[SoundEventSegment],
    ambience_beds: Sequence[AmbienceBed],
    physical_sources: Sequence[PhysicalSourceTrack],
    routing_judge: GeminiRoutingJudge | None,
) -> list[GenerationGroup]:
    events_by_id = {event.event_id: event for event in sound_events}
    ambience_by_id = {ambience.ambience_id: ambience for ambience in ambience_beds}
    sources_by_id = {source.source_id: source for source in physical_sources}
    updated: list[GenerationGroup] = []
    for group in generation_groups:
        if group.route_decision is not None and group.route_decision.decision_origin == "manual":
            updated.append(group)
            continue
        judgment = (
            routing_judge.judge_group(
                context={
                    "group_id": group.group_id,
                    "canonical_label": group.canonical_label,
                    "member_events": [
                        events_by_id[event_id].model_dump(mode="json")
                        for event_id in group.member_event_ids
                        if event_id in events_by_id
                    ],
                    "member_ambience": [
                        ambience_by_id[ambience_id].model_dump(mode="json")
                        for ambience_id in group.member_ambience_ids
                        if ambience_id in ambience_by_id
                    ],
                    "member_sources": [
                        sources_by_id[event.source_id].model_dump(mode="json")
                        for event in [events_by_id[event_id] for event_id in group.member_event_ids if event_id in events_by_id]
                        if event.source_id in sources_by_id
                    ],
                }
            )
            if routing_judge is not None
            else None
        )
        if judgment is None or judgment.model_type not in {"TTA", "VTA"}:
            updated.append(group)
            continue
        updated.append(
            group.model_copy(
                update={
                    "route_decision": RoutingDecision(
                        model_type=judgment.model_type,  # type: ignore[arg-type]
                        confidence=round(judgment.confidence, 4),
                        factors=[],
                        reasoning=judgment.reasoning,
                        decision_origin="gemini",
                    )
                }
            )
        )
    return updated
