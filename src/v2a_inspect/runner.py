from __future__ import annotations

from typing import Any, Callable, cast

from v2a_inspect.pipeline.response_models import GroupedAnalysis, VideoSceneAnalysis
from v2a_inspect.runtime import build_inspect_runtime
from v2a_inspect.workflows import (
    InspectOptions,
    InspectRuntime,
    InspectState,
    build_initial_inspect_state,
    build_inspect_graph,
    build_state_from_scene_analysis,
)
from v2a_inspect.workflows.inspect_graph import CompiledStateGraph

ProgressCallback = Callable[[str], None]


def run_inspect(
    video_path: str,
    *,
    options: InspectOptions | None = None,
    runtime: InspectRuntime | None = None,
    graph: CompiledStateGraph | None = None,
    progress_callback: ProgressCallback | None = None,
    warning_callback: ProgressCallback | None = None,
) -> InspectState:
    """Run the full inspect workflow for a video path."""

    resolved_options = options or InspectOptions()
    initial_state = build_initial_inspect_state(video_path, options=resolved_options)
    return _run_workflow(
        initial_state,
        runtime=runtime,
        graph=graph,
        options=resolved_options,
        progress_callback=progress_callback,
        warning_callback=warning_callback,
    )


def run_group_from_scene_analysis(
    scene_analysis: VideoSceneAnalysis,
    *,
    options: InspectOptions | None = None,
    runtime: InspectRuntime | None = None,
    graph: CompiledStateGraph | None = None,
    video_path: str = "",
    gemini_file: object | None = None,
    progress_callback: ProgressCallback | None = None,
    warning_callback: ProgressCallback | None = None,
) -> InspectState:
    """Run grouping and optional verification/model selection from scene JSON."""

    resolved_options = options or InspectOptions()
    initial_state = build_state_from_scene_analysis(
        scene_analysis,
        options=resolved_options,
        video_path=video_path,
        gemini_file=gemini_file,
    )
    return _run_workflow(
        initial_state,
        runtime=runtime,
        graph=graph,
        options=resolved_options,
        progress_callback=progress_callback,
        warning_callback=warning_callback,
    )


def get_grouped_analysis(state: InspectState) -> GroupedAnalysis:
    """Extract the final grouped analysis from workflow state."""

    grouped_analysis = state.get("grouped_analysis")
    if grouped_analysis is None:
        raise ValueError("Inspect workflow did not produce 'grouped_analysis'.")
    return grouped_analysis


def _run_workflow(
    initial_state: InspectState,
    *,
    options: InspectOptions,
    runtime: InspectRuntime | None,
    graph: CompiledStateGraph | None,
    progress_callback: ProgressCallback | None,
    warning_callback: ProgressCallback | None,
) -> InspectState:
    resolved_graph = graph or build_inspect_graph()
    graph_runner = cast(Any, resolved_graph)
    resolved_runtime = runtime or build_inspect_runtime(
        model=options.gemini_model,
        max_retries=options.max_retries,
    )

    last_state: dict[str, object] | None = None
    emitted_progress = 0
    emitted_warnings = 0

    for state_update in graph_runner.stream(
        initial_state,
        context=resolved_runtime,
        stream_mode="values",
    ):
        if not isinstance(state_update, dict):
            continue

        last_state = state_update

        progress_messages = list(state_update.get("progress_messages", []))
        if progress_callback is not None:
            while emitted_progress < len(progress_messages):
                progress_callback(progress_messages[emitted_progress])
                emitted_progress += 1
        else:
            emitted_progress = len(progress_messages)

        warnings = list(state_update.get("warnings", []))
        if warning_callback is not None:
            while emitted_warnings < len(warnings):
                warning_callback(warnings[emitted_warnings])
                emitted_warnings += 1
        else:
            emitted_warnings = len(warnings)

    if last_state is None:
        last_state = cast(
            dict[str, object],
            graph_runner.invoke(initial_state, context=resolved_runtime),
        )

    return cast(InspectState, last_state)
