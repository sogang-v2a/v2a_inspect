from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from v2a_inspect.llm import model_manager
from v2a_inspect.models import VideoAsset
from v2a_inspect.prompts.manager import PromptManager
from v2a_inspect.tools.sound_timeline import SoundTimelineEditor


def run_sound_timeline_agent(
    input_asset_path: Path,
    output_asset_path: Path | None = None,
    *,
    objective: str = "Build a complete SoundTimeline for the full video.",
    model: BaseChatModel | None = None,
    max_iterations: int = 80,
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
    max_iterations: int,
) -> dict:
    system_prompt, user_message = _prompt_messages(objective)
    agent = create_agent(
        model=model or model_manager.large,
        tools=editor.tools(),
        system_prompt=str(system_prompt.content),
        name="sound_timeline_agent",
    )
    return agent.invoke(
        {"messages": [HumanMessage(content=str(user_message.content))]},
        config={"recursion_limit": max_iterations},
    )


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
