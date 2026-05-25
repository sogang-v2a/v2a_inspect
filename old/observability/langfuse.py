from __future__ import annotations

import getpass
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from v2a_inspect.settings import settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    from langfuse.model import ChatMessageDict, ChatPromptClient
else:
    Langfuse = Any
    CallbackHandler = Any
    ChatMessageDict = dict[str, Any]
    ChatPromptClient = Any

LangfusePromptClient: TypeAlias = ChatPromptClient

_UNINITIALIZED = object()
_langfuse_client: Langfuse | None | object = _UNINITIALIZED


@dataclass(frozen=True)
class WorkflowTraceContext:
    source: Literal["cli", "ui", "runtime"]
    operation: Literal["analyze", "group"]
    user_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def is_langfuse_enabled() -> bool:
    return (
        settings.langfuse_public_key is not None
        and settings.langfuse_secret_key is not None
    )


def get_langfuse_client() -> Langfuse | None:
    global _langfuse_client

    if _langfuse_client is not _UNINITIALIZED:
        return cast("Langfuse | None", _langfuse_client)

    if not is_langfuse_enabled():
        _langfuse_client = None
        return None

    from langfuse import Langfuse

    _langfuse_client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value()
        if settings.langfuse_public_key is not None
        else None,
        secret_key=settings.langfuse_secret_key.get_secret_value()
        if settings.langfuse_secret_key is not None
        else None,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        release=get_release_name(),
        sample_rate=settings.langfuse_sample_rate,
    )
    return _langfuse_client


def require_langfuse_client() -> Langfuse:
    client = get_langfuse_client()
    if client is None:
        raise ValueError(
            "Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first."
        )
    return client


def get_release_name() -> str | None:
    if settings.langfuse_release:
        return settings.langfuse_release

    try:
        return version("v2a-inspect")
    except PackageNotFoundError:
        return None


def build_cli_trace_context(
    operation: Literal["analyze", "group"],
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> WorkflowTraceContext:
    return WorkflowTraceContext(
        source="cli",
        operation=operation,
        user_id=user_id or _safe_getpass_user(),
        session_id=session_id or uuid4().hex,
        tags=tuple(tags or []),
        metadata=dict(metadata or {}),
    )


def create_langfuse_handler(
    *,
    trace_id: str,
    parent_observation_id: str,
) -> CallbackHandler | None:
    if get_langfuse_client() is None:
        return None

    from langfuse.langchain import CallbackHandler

    return CallbackHandler(
        trace_context={
            "trace_id": trace_id,
            "parent_span_id": parent_observation_id,
        }
    )


def build_langgraph_runnable_config(
    *,
    handler: CallbackHandler | None,
    trace_context: WorkflowTraceContext,
    run_name: str,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig | None:
    if handler is None:
        return None

    combined_metadata = {
        "trace_source": trace_context.source,
        "trace_operation": trace_context.operation,
        **trace_context.metadata,
    }
    if metadata:
        combined_metadata.update(metadata)

    combined_tags = [
        "v2a-inspect",
        trace_context.source,
        trace_context.operation,
        *trace_context.tags,
    ]
    if tags:
        combined_tags.extend(tags)

    deduped_tags = list(dict.fromkeys(tag for tag in combined_tags if tag))
    return cast(
        RunnableConfig,
        {
            "callbacks": [handler],
            "run_name": run_name,
            "tags": deduped_tags,
            "metadata": combined_metadata,
        },
    )


def start_observation(**kwargs: Any) -> Any:
    client = get_langfuse_client()
    if client is None:
        return nullcontext(None)
    return client.start_as_current_observation(**kwargs)


def create_trace_score(
    *,
    trace_id: str,
    name: str,
    value: float | str,
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"],
    score_id: str | None = None,
    comment: str | None = None,
    metadata: Any = None,
    flush: bool = False,
) -> bool:
    client = get_langfuse_client()
    if client is None or not trace_id:
        return False

    if data_type == "CATEGORICAL":
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=str(value),
            data_type="CATEGORICAL",
            score_id=score_id,
            comment=comment,
            metadata=metadata,
        )
    else:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value),
            data_type=data_type,
            score_id=score_id,
            comment=comment,
            metadata=metadata,
        )
    if flush:
        client.flush()
    return True


def build_score_id(trace_id: str, name: str, *parts: str) -> str:
    return ":".join([trace_id, name, *parts])


def sync_chat_prompt(
    *,
    name: str,
    system_prompt: str,
    user_prompt: str,
    label: str | None = None,
) -> LangfusePromptClient:
    client = require_langfuse_client()
    labels = [label or settings.langfuse_prompt_label]
    prompt_messages: list[ChatMessageDict] = []
    if system_prompt.strip():
        prompt_messages.append(ChatMessageDict(role="system", content=system_prompt))
    prompt_messages.append(ChatMessageDict(role="user", content=user_prompt))

    return client.create_prompt(
        name=name,
        prompt=cast(Any, prompt_messages),
        type="chat",
        labels=labels,
    )


def fetch_chat_prompt(
    name: str,
    *,
    label: str | None = None,
    fallback: list[ChatMessageDict] | None = None,
) -> LangfusePromptClient | None:
    client = get_langfuse_client()
    if client is None:
        return None

    return client.get_prompt(
        name,
        label=label or settings.langfuse_prompt_label,
        type="chat",
        fallback=fallback,
    )


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is not None:
        client.flush()


def _safe_getpass_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown-user"
