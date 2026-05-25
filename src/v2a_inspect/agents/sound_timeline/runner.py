from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from v2a_inspect.config import settings
from v2a_inspect.llm import model_manager
from v2a_inspect.models import VideoAsset
from v2a_inspect.observability import build_langchain_config, flush_langfuse
from v2a_inspect.prompts.manager import PromptManager
from v2a_inspect.tools.sound_timeline import SoundTimelineEditor


def run_sound_timeline_agent(
    input_asset_path: Path,
    output_asset_path: Path | None = None,
    *,
    objective: str = "Build a complete SoundTimeline for the full video.",
    model: BaseChatModel | None = None,
    max_iterations: int | None = None,
) -> VideoAsset:
    video_asset = VideoAsset.model_validate_json(input_asset_path.read_text())
    editor = SoundTimelineEditor(video_asset)
    run_agent_loop(
        editor, objective=objective, model=model, max_iterations=max_iterations
    )
    if output_asset_path is None:
        output_asset_path = input_asset_path.with_name(
            f"{input_asset_path.stem}.with_sound_timeline{input_asset_path.suffix}"
        )
    output_asset_path.write_text(
        editor.video_asset.model_dump_json(indent=2, exclude_computed_fields=True),
        encoding="utf-8",
    )
    return editor.video_asset


def run_agent_loop(
    editor: SoundTimelineEditor,
    *,
    objective: str,
    model: BaseChatModel | None,
    max_iterations: int | None,
) -> dict:
    resolved_max_iterations = _resolve_max_iterations(max_iterations)
    system_prompt, user_message = _prompt_messages(objective)
    agent = create_agent(
        model=model or model_manager.large,
        tools=editor.tools(),
        system_prompt=str(system_prompt.content),
        name="sound_timeline_agent",
    )
    try:
        return agent.invoke(
            {"messages": [HumanMessage(content=str(user_message.content))]},
            config=_agent_invoke_config(
                editor,
                objective=objective,
                max_iterations=resolved_max_iterations,
            ),
        )
    finally:
        flush_langfuse()


def _resolve_max_iterations(max_iterations: int | None) -> int:
    if max_iterations is None:
        return settings.agent_sound_timeline_recursion_limit
    if max_iterations < 1:
        raise ValueError("max_iterations must be greater than or equal to 1")
    return max_iterations


def _agent_invoke_config(
    editor: SoundTimelineEditor,
    *,
    objective: str,
    max_iterations: int,
) -> RunnableConfig:
    config = RunnableConfig(recursion_limit=max_iterations)
    trace_config = build_langchain_config(
        run_name="sound_timeline_agent",
        tags=["v2a-inspect", "agent", "sound-timeline"],
        metadata=_trace_metadata(
            editor,
            objective=objective,
            max_iterations=max_iterations,
        ),
    )
    if trace_config is not None:
        config.update(trace_config)
    return config


def _trace_metadata(
    editor: SoundTimelineEditor,
    *,
    objective: str,
    max_iterations: int,
) -> dict[str, object]:
    video_asset = editor.video_asset
    timeline = video_asset.sound_timeline
    return {
        "video_id": str(video_asset.video_id),
        "source_path": str(video_asset.source_path),
        "frame_count": video_asset.frame_count,
        "initial_scene_count": len(video_asset.initial_scenes),
        "sound_source_count": 0 if timeline is None else len(timeline.sound_sources),
        "sound_event_count": 0 if timeline is None else len(timeline.sound_events),
        "objective": objective,
        "max_iterations": max_iterations,
    }


def _prompt_messages(objective: str):
    messages = (
        PromptManager()
        .get_prompt("sound_timeline_agent")
        .format_messages(objective=objective)
    )
    if len(messages) != 2:
        raise ValueError(
            "sound_timeline_agent prompt must have system and user messages"
        )
    return messages[0], messages[1]
