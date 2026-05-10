from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .langfuse import (
        LangfusePromptClient,
        WorkflowTraceContext,
        build_cli_trace_context,
        build_langgraph_runnable_config,
        build_score_id,
        create_langfuse_handler,
        create_trace_score,
        fetch_chat_prompt,
        flush_langfuse,
        get_langfuse_client,
        get_release_name,
        is_langfuse_enabled,
        require_langfuse_client,
        start_observation,
        sync_chat_prompt,
    )

__all__ = [
    "LangfusePromptClient",
    "WorkflowTraceContext",
    "build_cli_trace_context",
    "build_langgraph_runnable_config",
    "build_score_id",
    "create_langfuse_handler",
    "create_trace_score",
    "fetch_chat_prompt",
    "flush_langfuse",
    "get_langfuse_client",
    "get_release_name",
    "is_langfuse_enabled",
    "require_langfuse_client",
    "start_observation",
    "sync_chat_prompt",
]

_EXPORT_MAP = {
    name: ("v2a_inspect.observability.langfuse", name) for name in __all__
}


def __getattr__(name: str) -> Any:
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
