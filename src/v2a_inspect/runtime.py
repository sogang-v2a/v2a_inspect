from __future__ import annotations

from dataclasses import dataclass

import google.genai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from v2a_inspect.clients import DEFAULT_GEMINI_MODEL
from v2a_inspect.tools import (
    EmbeddingRunpodClient,
    RemoteGpuPolicy,
    RemoteGpuSelection,
    Sam3RunpodClient,
    Siglip2LabelClient,
    choose_remote_gpu,
)
from v2a_inspect.workflows import InspectRuntime


def build_genai_client(*, api_key: str | None = None) -> genai.Client:
    """Build a Gemini SDK client for file uploads and file lookup."""

    return genai.Client(api_key=api_key or _require_gemini_api_key())


def build_llm(
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Build the LangChain Gemini chat model used by the workflow."""

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key or _require_gemini_api_key(),
        max_retries=max(1, max_retries),
        timeout=timeout_seconds,
    )


@dataclass(frozen=True)
class ToolingRuntime:
    gpu_selection: RemoteGpuSelection
    sam3_client: Sam3RunpodClient | None
    embedding_client: EmbeddingRunpodClient | None
    label_client: Siglip2LabelClient | None


def build_inspect_runtime(
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
    llm: ChatGoogleGenerativeAI | None = None,
    genai_client: genai.Client | None = None,
) -> InspectRuntime:
    """Build workflow runtime dependencies for the inspect graph."""

    resolved_api_key = api_key or _require_gemini_api_key()
    return InspectRuntime(
        llm=llm
        if llm is not None
        else build_llm(
            model=model,
            api_key=resolved_api_key,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        ),
        genai_client=genai_client
        if genai_client is not None
        else build_genai_client(api_key=resolved_api_key),
    )


def build_tooling_runtime() -> ToolingRuntime:
    from v2a_inspect.settings import settings

    gpu_policy = RemoteGpuPolicy(
        preferred_sku=settings.remote_gpu_preference,
        fallback_sku=settings.remote_gpu_fallback,
        preferred_vram_gb=settings.remote_gpu_vram_preference_gb,
        max_vram_gb=settings.remote_gpu_vram_cap_gb,
    )
    api_key = (
        settings.runpod_api_key.get_secret_value()
        if settings.runpod_api_key is not None
        else None
    )
    timeout = settings.remote_timeout_seconds
    return ToolingRuntime(
        gpu_selection=choose_remote_gpu(gpu_policy),
        sam3_client=(
            Sam3RunpodClient(
                endpoint_url=settings.sam3_endpoint_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )
            if settings.sam3_endpoint_url
            else None
        ),
        embedding_client=(
            EmbeddingRunpodClient(
                endpoint_url=settings.embedding_endpoint_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )
            if settings.embedding_endpoint_url
            else None
        ),
        label_client=(
            Siglip2LabelClient(
                endpoint_url=settings.label_endpoint_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )
            if settings.label_endpoint_url
            else None
        ),
    )


def _require_gemini_api_key() -> str:
    from v2a_inspect.settings import settings

    if settings.gemini_api_key is not None:
        return settings.gemini_api_key.get_secret_value()

    if settings.openrouter_api_key is not None:
        raise ValueError(
            "GEMINI_API_KEY must be set for the inspect runtime. "
            "OpenRouter alone is not enough because Gemini file upload is required."
        )

    raise ValueError(
        "GEMINI_API_KEY must be set for the inspect runtime. "
        "The workflow uses Gemini file uploads plus ChatGoogleGenerativeAI."
    )
