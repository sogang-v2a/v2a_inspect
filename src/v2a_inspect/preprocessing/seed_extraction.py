from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from v2a_inspect.llm import model_manager
from v2a_inspect.models import InitialScene, InitialSceneAnalysis, Keyframe
from v2a_inspect.observability import build_langchain_config
from v2a_inspect.prompts.manager import PromptManager


def analyze_initial_scene(
    initial_scene: InitialScene,
    model: BaseChatModel | None = None,
) -> InitialScene:
    prompt = PromptManager().get_multimodal_prompt(
        "initial_scene_analysis",
        content_blocks=_image_blocks(initial_scene.keyframes),
    )
    chat_model = model or model_manager.large
    chain = prompt | chat_model.with_structured_output(InitialSceneAnalysis)
    result = chain.invoke(
        _scene_inputs(initial_scene),
        config=_trace_config(initial_scene),
    )
    analysis = InitialSceneAnalysis.model_validate(result)
    return initial_scene.model_copy(update={"initial_analysis": analysis})


def analyze_initial_scenes(
    initial_scenes: list[InitialScene],
    model: BaseChatModel | None = None,
) -> list[InitialScene]:
    analyzed_scenes: list[InitialScene] = []
    for initial_scene in initial_scenes:
        analyzed_scenes.append(analyze_initial_scene(initial_scene, model=model))
    return analyzed_scenes


def _keyframe_indexes_text(keyframes: list[Keyframe]) -> str:
    frame_indexes: list[str] = []
    for frame_index in _keyframe_indexes(keyframes):
        frame_indexes.append(str(frame_index))
    return ", ".join(frame_indexes)


def _keyframe_indexes(keyframes: list[Keyframe]) -> list[int]:
    frame_indexes: list[int] = []
    for keyframe in keyframes:
        frame_indexes.append(keyframe.frame_index)
    return frame_indexes


def _scene_inputs(initial_scene: InitialScene) -> dict[str, Any]:
    return {
        "initial_scene_id": initial_scene.initial_scene_id,
        "start_frame_index": initial_scene.start_frame_index,
        "end_frame_index": initial_scene.end_frame_index,
        "frame_count": initial_scene.frame_count,
        "keyframe_indexes": _keyframe_indexes_text(initial_scene.keyframes),
    }


def _trace_config(initial_scene: InitialScene) -> RunnableConfig | None:
    return build_langchain_config(
        run_name="initial_scene_analysis",
        tags=["v2a-inspect", "preprocessing", "initial-scene-analysis"],
        metadata={
            "initial_scene_id": str(initial_scene.initial_scene_id),
            "start_frame_index": str(initial_scene.start_frame_index),
            "end_frame_index": str(initial_scene.end_frame_index),
            "frame_count": str(initial_scene.frame_count),
            "keyframe_count": str(len(initial_scene.keyframes)),
            "keyframe_indexes": _keyframe_indexes_text(initial_scene.keyframes),
        },
    )


def _image_blocks(keyframes: list[Keyframe]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for keyframe in keyframes:
        blocks.append(_image_block(keyframe.image_path))
    return blocks


def _image_block(image_path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }
