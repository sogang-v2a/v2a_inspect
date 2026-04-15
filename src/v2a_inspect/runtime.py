from __future__ import annotations

import os

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.openai_compat_llm import build_openai_compat_llm


def build_llm(
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
) -> object:
    """Build the Gemini chat model used by the server-side silent-video pipeline."""

    openai_compat_base_url = os.getenv("V2A_LLM_BASE_URL", "").strip()
    if openai_compat_base_url:
        resolved_model = os.getenv("V2A_LLM_MODEL", "").strip() or model
        openai_compat_api_key = os.getenv("V2A_LLM_API_KEY", "").strip() or api_key
        return build_openai_compat_llm(
            model=resolved_model,
            base_url=openai_compat_base_url,
            api_key=openai_compat_api_key or None,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

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
