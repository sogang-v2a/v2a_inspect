from __future__ import annotations

import os
import json
import re
from typing import Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from v2a_inspect.constants import DEFAULT_GEMINI_MODEL


StructuredSchemaModel = TypeVar("StructuredSchemaModel")


class StructuredInvoker(Protocol):
    def invoke(self, prompt: object) -> object: ...


class StructuredChatModel(Protocol):
    def invoke(self, prompt: object) -> object: ...

    def with_structured_output(
        self,
        schema_model: type[StructuredSchemaModel],
        *,
        method: str = "json_schema",
    ) -> StructuredInvoker: ...


def build_llm(
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
) -> StructuredChatModel:
    """Build the chat model used by the silent-video semantic pipeline."""

    openai_compat_base_url = os.getenv("V2A_LLM_BASE_URL", "").strip()
    if openai_compat_base_url:
        from langchain_openai import ChatOpenAI

        resolved_model = os.getenv("V2A_LLM_MODEL", "").strip() or model
        openai_compat_api_key = os.getenv("V2A_LLM_API_KEY", "").strip()
        return ChatOpenAI(
            model=resolved_model,
            base_url=openai_compat_base_url,
            api_key=openai_compat_api_key or "unused",
            max_retries=max(1, max_retries),
            timeout=timeout_seconds,
            use_responses_api=False,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key or _require_gemini_api_key(),
        max_retries=max(1, max_retries),
        timeout=timeout_seconds,
    )


def invoke_structured_llm(
    *,
    llm: StructuredChatModel,
    schema_model: type[StructuredSchemaModel],
    prompt: object,
    method: str = "json_schema",
) -> StructuredSchemaModel:
    if _is_chatopenai_instance(llm):
        response = llm.invoke(_augment_prompt_with_json_schema(prompt, schema_model))
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            raise TypeError("ChatOpenAI response content must be a string.")
        parsed = _extract_json_object(_strip_think_blocks(content).strip())
        model_validate = getattr(schema_model, "model_validate", None)
        if callable(model_validate):
            return model_validate(parsed)
        return parsed  # type: ignore[return-value]

    structured_llm = llm.with_structured_output(
        schema_model,
        method=method,
    )
    result = structured_llm.invoke(prompt)
    if isinstance(result, schema_model):
        return result
    model_validate = getattr(schema_model, "model_validate", None)
    if callable(model_validate):
        return model_validate(result)
    return result  # type: ignore[return-value]


def _require_gemini_api_key() -> str:
    from v2a_inspect.settings import settings

    if settings.gemini_api_key is not None:
        return settings.gemini_api_key.get_secret_value()

    raise ValueError(
        "GEMINI_API_KEY must be set for the server-side silent-video Gemini helpers."
    )


def _is_chatopenai_instance(llm: object) -> bool:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return False
    return isinstance(llm, ChatOpenAI)


def _augment_prompt_with_json_schema(
    prompt: object,
    schema_model: type[object],
) -> list[object]:
    schema = _model_json_schema(schema_model)
    schema_instruction = SystemMessage(
        content=(
            "Respond with only one JSON object that matches this schema exactly. "
            "Do not include commentary, markdown, or explanation.\n\n"
            f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
        )
    )
    if isinstance(prompt, list):
        return [schema_instruction, *prompt]
    return [
        schema_instruction,
        HumanMessage(content=str(prompt)),
    ]


def _model_json_schema(schema_model: type[object]) -> dict[str, object]:
    model_json_schema = getattr(schema_model, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    raise TypeError("schema_model must provide model_json_schema().")


def _strip_think_blocks(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)


def _extract_json_object(content: str) -> dict[str, object]:
    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Structured ChatOpenAI response did not contain a JSON object.")
