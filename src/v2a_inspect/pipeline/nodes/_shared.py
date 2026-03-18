from __future__ import annotations

from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from v2a_inspect.clients import build_uploaded_video_content_block

from ..prompt_templates import (
    SCENE_ANALYSIS_DEFAULT_PROMPT_TEMPLATE,
    SCENE_ANALYSIS_EXTENDED_PROMPT_TEMPLATE,
)
from ..response_models import RawTrack, TrackGroup
from v2a_inspect.workflows.state import InspectOptions, InspectState

T = TypeVar("T", bound=BaseModel)


def append_state_message(
    state: InspectState,
    key: Literal["errors", "warnings", "progress_messages"],
    message: str,
) -> list[str]:
    return [*state.get(key, []), message]


def get_scene_analysis_prompt(options: InspectOptions) -> str:
    if options.scene_analysis_mode == "extended":
        return SCENE_ANALYSIS_EXTENDED_PROMPT_TEMPLATE
    return SCENE_ANALYSIS_DEFAULT_PROMPT_TEMPLATE


def build_text_message(prompt: str) -> HumanMessage:
    return HumanMessage(content=[{"type": "text", "text": prompt}])


def build_video_message(file_obj: Any, *, fps: float, prompt: str) -> HumanMessage:
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            build_uploaded_video_content_block(file_obj, fps=fps),
        ]
    )


def configure_llm(
    llm: BaseChatModel,
    *,
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    updates: dict[str, Any] = {}

    if model:
        updates["model"] = model
    if timeout_ms is not None:
        updates["timeout"] = timeout_ms / 1000
    if max_retries is not None:
        updates["max_retries"] = max(1, max_retries)

    if not updates:
        return llm

    model_copy = getattr(llm, "model_copy", None)
    if not callable(model_copy):
        raise TypeError(
            "Inspect workflow requires a Pydantic-based chat model for per-call overrides."
        )

    configured_llm = model_copy(update=updates)
    if not isinstance(configured_llm, BaseChatModel):
        raise TypeError("Inspect workflow requires a BaseChatModel runtime dependency.")
    return configured_llm


def invoke_structured_text(
    llm: BaseChatModel,
    *,
    prompt: str,
    schema: type[T],
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    label: str = "",
) -> T:
    return _invoke_structured(
        llm=llm,
        message=build_text_message(prompt),
        schema=schema,
        model=model,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        label=label,
    )


def invoke_structured_video(
    llm: BaseChatModel,
    *,
    file_obj: Any,
    fps: float,
    prompt: str,
    schema: type[T],
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    label: str = "",
) -> T:
    return _invoke_structured(
        llm=llm,
        message=build_video_message(file_obj, fps=fps, prompt=prompt),
        schema=schema,
        model=model,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        label=label,
    )


def _invoke_structured(
    llm: BaseChatModel,
    *,
    message: HumanMessage,
    schema: type[T],
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    label: str = "",
) -> T:
    configured_llm = configure_llm(
        llm,
        model=model,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
    )
    structured_llm = configured_llm.with_structured_output(
        schema,
        method="json_schema",
    )

    try:
        result = structured_llm.invoke([message])
    except Exception as exc:  # noqa: BLE001
        error_label = f" for {label}" if label else ""
        raise RuntimeError(f"LLM request failed{error_label}: {exc}") from exc

    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def get_active_groups(state: InspectState) -> list[TrackGroup]:
    final_groups = state.get("final_groups")
    if final_groups is not None:
        return list(final_groups)

    verified_groups = state.get("verified_groups")
    if verified_groups is not None:
        return list(verified_groups)

    text_groups = state.get("text_groups")
    if text_groups is not None:
        return list(text_groups)

    return []


def build_grouping_numbered_list(tracks: list[RawTrack]) -> str:
    return "\n".join(
        f"[{index}] {track.track_id} ({track.kind}, scene {track.scene_index}, "
        f"{track.start:.1f}s-{track.end:.1f}s): {track.description}"
        for index, track in enumerate(tracks)
    )


def build_verify_segment_list(
    group: TrackGroup,
    tracks_by_id: dict[str, RawTrack],
) -> str:
    lines: list[str] = []
    for index, track_id in enumerate(group.member_ids):
        track = tracks_by_id[track_id]
        lines.append(
            f"  Segment {index}: scene {track.scene_index}, "
            f'{track.start:.1f}s-{track.end:.1f}s | "{track.description}"'
        )
    return "\n".join(lines)


def build_model_select_segment_list(member_tracks: list[RawTrack]) -> str:
    return "\n".join(
        f"  Segment {index}: scene {track.scene_index}, {track.start:.1f}s-{track.end:.1f}s"
        f" | kind={track.kind} | n_objects_in_scene={track.n_scene_objects}"
        f' | "{track.description}"'
        for index, track in enumerate(member_tracks)
    )
