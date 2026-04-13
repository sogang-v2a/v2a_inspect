from __future__ import annotations

from typing import TYPE_CHECKING

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI


def build_llm(
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Build the Gemini chat model used by the server-side silent-video pipeline."""

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key or _require_gemini_api_key(),
        max_retries=max(1, max_retries),
        timeout=timeout_seconds,
    )


def _require_gemini_api_key() -> str:
    from v2a_inspect.settings import settings

    if settings.gemini_api_key is not None:
        return settings.gemini_api_key.get_secret_value()

    raise ValueError(
        "GEMINI_API_KEY must be set for the server-side silent-video Gemini helpers."
    )
