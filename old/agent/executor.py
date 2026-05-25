from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, runtime_checkable
from uuid import uuid4

from v2a_inspect.agent.state import (
    DecisionRecord,
    PlannedAction,
    PlannerState,
    ToolCallRecord,
)


@dataclass(frozen=True)
class ToolExecutor:
    registry: dict[str, Callable[..., object]]
    trace_path: Path | None = None

    def execute(
        self, state: PlannerState, action: PlannedAction
    ) -> tuple[PlannerState, object]:
        handler = self.registry[action.tool_name]
        effective_request_payload, dropped_request_keys = _filter_payload_for_handler(
            handler,
            action.request_payload,
        )
        result = handler(**effective_request_payload)
        record = ToolCallRecord(
            call_id=uuid4().hex,
            issue_id=action.issue_id,
            tool_name=action.tool_name,
            request_payload=_sanitize_payload(action.request_payload),
            effective_request_payload=_sanitize_payload(effective_request_payload),
            dropped_request_keys=dropped_request_keys,
            output_refs=_extract_output_refs(result),
        )
        updated = state.model_copy(deep=True)
        updated.tool_calls.append(record)
        if self.trace_path is not None:
            self._append_trace({"kind": "tool_call", **record.model_dump(mode="json")})
        return updated, result

    def record_decision(
        self,
        state: PlannerState,
        *,
        issue_id: str,
        decision: Literal["accept", "reject", "retry", "skip"],
        rationale: str,
        confidence: float,
    ) -> PlannerState:
        updated = state.model_copy(deep=True)
        record = DecisionRecord(
            issue_id=issue_id,
            decision=decision,
            rationale=rationale,
            confidence=confidence,
        )
        updated.decisions.append(record)
        if self.trace_path is not None:
            self._append_trace({"kind": "decision", **record.model_dump(mode="json")})
        return updated

    def replay_trace(self) -> list[dict[str, object]]:
        if self.trace_path is None or not self.trace_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _append_trace(self, payload: dict[str, object]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(payload) + "\n")


@runtime_checkable
class _Dumpable(Protocol):
    def model_dump(self, *, mode: str = "json") -> dict[str, object]: ...


def _extract_output_refs(result: object) -> list[str]:
    refs: list[str] = []
    if isinstance(result, dict):
        for key, value in result.items():
            if (
                isinstance(key, str)
                and key.endswith("_path")
                and isinstance(value, str)
            ):
                refs.append(value)
            if (
                isinstance(key, str)
                and key.endswith("_ids")
                and isinstance(value, list)
            ):
                refs.extend(str(item) for item in value)
    elif isinstance(result, _Dumpable):
        return _extract_output_refs(result.model_dump(mode="json"))
    return refs


def _sanitize_payload(payload: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [str(item) for item in value[:10]]
        elif isinstance(value, dict):
            sanitized[key] = {str(item_key): str(item_value) for item_key, item_value in list(value.items())[:10]}
        elif isinstance(value, _Dumpable):
            sanitized[key] = value.model_dump(mode="json")
        else:
            sanitized[key] = repr(value)
    return sanitized


def _filter_payload_for_handler(
    handler: Callable[..., object],
    payload: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    signature = inspect.signature(handler)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(payload), []

    accepted_keys = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    effective = {
        key: value for key, value in payload.items() if key in accepted_keys
    }
    dropped = sorted(key for key in payload if key not in accepted_keys)
    return effective, dropped
