from __future__ import annotations

from typing import TYPE_CHECKING

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.workflows import InspectRuntime

if TYPE_CHECKING:
    import google.genai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI


def build_genai_client(*, api_key: str | None = None) -> genai.Client:
    """Build a Gemini SDK client for file uploads and file lookup."""

    import google.genai as genai

    return genai.Client(api_key=api_key or _require_gemini_api_key())


def build_llm(
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Build the LangChain Gemini chat model used by the workflow."""

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key or _require_gemini_api_key(),
        max_retries=max(1, max_retries),
        timeout=timeout_seconds,
    )


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
