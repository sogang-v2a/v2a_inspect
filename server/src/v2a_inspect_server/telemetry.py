from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from v2a_inspect.workflows import InspectState


def stage_start() -> float:
    return monotonic()


def ensure_runtime_trace_path(state: InspectState) -> str | None:
    existing = state.get("runtime_trace_path")
    if isinstance(existing, str) and existing:
        return existing
    run_dir = state.get("artifact_run_dir")
    video_path = state.get("video_path")
    if not isinstance(run_dir, str) or not run_dir or not isinstance(video_path, str):
        return None
    trace_path = str(Path(run_dir) / f"{Path(video_path).stem or 'video'}-runtime-trace.jsonl")
    state["runtime_trace_path"] = trace_path
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return trace_path


def record_stage(
    state: InspectState,
    *,
    stage: str,
    started_at: float,
    metrics: dict[str, object] | None = None,
) -> None:
    event = {
        "kind": "stage",
        "stage": stage,
        "elapsed_seconds": round(monotonic() - started_at, 4),
        "recorded_at": datetime.now(UTC).isoformat(),
        **(metrics or {}),
    }
    history = list(state.get("stage_history", []))
    history.append(event)
    state["stage_history"] = history
    _append_trace(state, event)


def record_recovery_attempt(
    state: InspectState,
    *,
    tool_name: str,
    details: dict[str, object] | None = None,
) -> None:
    entry = {
        "tool_name": tool_name,
        "recorded_at": datetime.now(UTC).isoformat(),
        **(details or {}),
    }
    attempts = list(state.get("recovery_attempts", []))
    attempts.append(entry)
    state["recovery_attempts"] = attempts
    _append_trace(state, {"kind": "recovery_attempt", **entry})


def _append_trace(state: InspectState, payload: dict[str, object]) -> None:
    trace_path = ensure_runtime_trace_path(state)
    if trace_path is None:
        return
    with Path(trace_path).open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload) + "\n")
