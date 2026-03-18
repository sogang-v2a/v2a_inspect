from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence, cast

import google.genai as genai
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from v2a_inspect.pipeline.nodes import (
    analyze_scenes,
    assemble_grouped_analysis,
    extract_raw_tracks,
    group_tracks,
    select_models,
    upload_video,
    verify_groups,
)
from v2a_inspect.pipeline.response_models import VideoSceneAnalysis

from .state import InspectOptions, InspectState


@dataclass(frozen=True)
class InspectRuntime:
    """Runtime dependencies for the inspect workflow."""

    llm: BaseChatModel
    genai_client: genai.Client


def build_initial_inspect_state(
    video_path: str,
    *,
    options: InspectOptions | None = None,
) -> InspectState:
    """Build a fresh initial state for full video inspection."""

    return InspectState(
        video_path=video_path,
        options=options or InspectOptions(),
        errors=[],
        warnings=[],
        progress_messages=[],
    )


def build_state_from_scene_analysis(
    scene_analysis: VideoSceneAnalysis,
    *,
    options: InspectOptions | None = None,
    video_path: str = "",
    gemini_file: Any | None = None,
) -> InspectState:
    """Build state seeded from scene analysis for future grouping-only flows."""

    state = InspectState(
        scene_analysis=scene_analysis,
        options=options or InspectOptions(),
        errors=[],
        warnings=[],
        progress_messages=[],
    )
    if video_path:
        state["video_path"] = video_path
    if gemini_file is not None:
        state["gemini_file"] = gemini_file
    return state


def build_inspect_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
) -> CompiledStateGraph:
    """Build the compiled LangGraph workflow for the inspect pipeline."""

    graph = cast(Any, StateGraph)(InspectState, context_schema=InspectRuntime)

    graph.add_node("bootstrap", _bootstrap_node)
    graph.add_node("upload", _upload_node)
    graph.add_node("analyze", _analyze_node)
    graph.add_node("extract", _extract_node)
    graph.add_node("group", _group_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("select_model", _select_model_node)
    graph.add_node("assemble", _assemble_node)

    graph.add_edge(START, "bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        _route_after_bootstrap,
        {
            "upload": "upload",
            "analyze": "analyze",
            "extract": "extract",
        },
    )
    graph.add_conditional_edges(
        "upload",
        _route_after_upload,
        {
            "analyze": "analyze",
            "extract": "extract",
        },
    )
    graph.add_edge("analyze", "extract")
    graph.add_edge("extract", "group")
    graph.add_conditional_edges(
        "group",
        _route_after_group,
        {
            "verify": "verify",
            "select_model": "select_model",
            "assemble": "assemble",
        },
    )
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "select_model": "select_model",
            "assemble": "assemble",
        },
    )
    graph.add_edge("select_model", "assemble")
    graph.add_edge("assemble", END)

    return cast(
        CompiledStateGraph,
        graph.compile(
            checkpointer=checkpointer,
            interrupt_before=list(interrupt_before)
            if interrupt_before is not None
            else None,
            interrupt_after=list(interrupt_after)
            if interrupt_after is not None
            else None,
            debug=debug,
            name="v2a_inspect_workflow",
        ),
    )


def _upload_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
) -> dict[str, object]:
    return _run_node(
        "upload",
        lambda: upload_video(state, genai_client=runtime.context.genai_client),
    )


def _bootstrap_node(state: InspectState) -> dict[str, object]:
    return cast(dict[str, object], state)


def _analyze_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
) -> dict[str, object]:
    return _run_node(
        "analyze",
        lambda: analyze_scenes(state, llm=runtime.context.llm),
    )


def _extract_node(state: InspectState) -> dict[str, object]:
    return _run_node("extract", lambda: extract_raw_tracks(state))


def _group_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
) -> dict[str, object]:
    return _run_node(
        "group",
        lambda: group_tracks(state, llm=runtime.context.llm),
    )


def _verify_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
) -> dict[str, object]:
    return _run_node(
        "verify",
        lambda: verify_groups(state, llm=runtime.context.llm),
    )


def _select_model_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
) -> dict[str, object]:
    return _run_node(
        "select_model",
        lambda: select_models(state, llm=runtime.context.llm),
    )


def _assemble_node(state: InspectState) -> dict[str, object]:
    return _run_node("assemble", lambda: assemble_grouped_analysis(state))


def _route_after_group(
    state: InspectState,
) -> Literal["verify", "select_model", "assemble"]:
    options = _get_options(state)
    if options.enable_vlm_verify:
        return "verify"
    if options.enable_model_select:
        return "select_model"
    return "assemble"


def _route_after_bootstrap(
    state: InspectState,
) -> Literal["upload", "analyze", "extract"]:
    if state.get("scene_analysis") is not None:
        if state.get("gemini_file") is not None:
            return "extract"
        if _requires_video_context(state) and state.get("video_path"):
            return "upload"
        return "extract"

    if state.get("gemini_file") is not None:
        return "analyze"
    return "upload"


def _route_after_upload(state: InspectState) -> Literal["analyze", "extract"]:
    if state.get("scene_analysis") is not None:
        return "extract"
    return "analyze"


def _route_after_verify(
    state: InspectState,
) -> Literal["select_model", "assemble"]:
    if _get_options(state).enable_model_select:
        return "select_model"
    return "assemble"


def _get_options(state: InspectState) -> InspectOptions:
    options = state.get("options")
    if options is None:
        raise ValueError("inspect graph state is missing 'options'.")
    return options


def _requires_video_context(state: InspectState) -> bool:
    options = _get_options(state)
    return options.enable_vlm_verify or options.enable_model_select


def _run_node(
    node_name: str,
    action: Callable[[], dict[str, object]],
) -> dict[str, object]:
    try:
        return action()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"inspect graph failed in '{node_name}': {exc}") from exc
