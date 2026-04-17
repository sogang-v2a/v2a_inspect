from __future__ import annotations

import json
import re
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def build_openai_compat_llm(
    *,
    model: str,
    base_url: str,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: float | None = None,
) -> "_OpenAICompatChatModel":
    return _OpenAICompatChatModel(
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        max_retries=max(1, max_retries),
        timeout_seconds=timeout_seconds or 60.0,
    )


@dataclass(frozen=True)
class _OpenAICompatChatModel:
    model: str
    base_url: str
    api_key: str | None
    max_retries: int
    timeout_seconds: float

    def invoke(self, prompt: object) -> str:
        payload = {
            "model": self.model,
            "messages": _normalize_messages(prompt),
            "max_tokens": 2048,
        }
        response = _post_json(
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        return _extract_text_content(response)

    def with_structured_output(
        self,
        schema_model: type[object],
        *,
        method: str = "json_schema",
    ) -> "_OpenAICompatStructuredModel":
        return _OpenAICompatStructuredModel(
            chat_model=self,
            schema_model=schema_model,
            method=method,
        )


@dataclass(frozen=True)
class _OpenAICompatStructuredModel:
    chat_model: _OpenAICompatChatModel
    schema_model: type[object]
    method: str

    def invoke(self, prompt: object) -> object:
        schema = _model_json_schema(self.schema_model)
        messages = _normalize_messages(prompt)
        messages = [
            {
                "role": "system",
                "content": (
                    "Respond with only one JSON object that matches this schema exactly. "
                    "Do not include commentary, markdown, or explanation.\n\n"
                    f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
                ),
            },
            *messages,
        ]
        payload = {
            "model": self.chat_model.model,
            "messages": messages,
            "max_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(self.schema_model, "__name__", "structured_response"),
                    "schema": schema,
                },
            },
        }
        response = _post_json(
            url=f"{self.chat_model.base_url}/chat/completions",
            payload=payload,
            api_key=self.chat_model.api_key,
            timeout_seconds=self.chat_model.timeout_seconds,
            max_retries=self.chat_model.max_retries,
        )
        content = _extract_text_content(response)
        parsed = _extract_json_object(content)
        model_validate = getattr(self.schema_model, "model_validate", None)
        if callable(model_validate):
            return model_validate(parsed)
        return parsed


def _post_json(
    *,
    url: str,
    payload: Mapping[str, object],
    api_key: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise TypeError("OpenAI-compatible response must be a JSON object.")
                return decoded
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 >= max_retries:
                raise
            time.sleep(min(0.5 * (attempt + 1), 2.0))
    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenAI-compatible request failed without an exception.")


def _normalize_messages(prompt: object) -> list[dict[str, object]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list):
        return [{"role": "user", "content": repr(prompt)}]
    messages: list[dict[str, object]] = []
    for item in prompt:
        role = _message_role(item)
        content = getattr(item, "content", item)
        if isinstance(content, list):
            normalized_content: list[dict[str, object]] = []
            for block in content:
                if isinstance(block, dict):
                    normalized_content.append(
                        {str(key): value for key, value in block.items()}
                    )
                else:
                    normalized_content.append({"type": "text", "text": str(block)})
            messages.append({"role": role, "content": normalized_content})
        else:
            messages.append({"role": role, "content": str(content)})
    return messages


def _message_role(message: object) -> str:
    explicit = getattr(message, "type", None)
    if explicit == "system":
        return "system"
    if explicit == "ai":
        return "assistant"
    return "user"


def _extract_text_content(response_payload: Mapping[str, object]) -> str:
    choices = response_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI-compatible response is missing choices.")
    first_choice = choices[0]
    choice_payload = (
        {str(key): value for key, value in first_choice.items()}
        if isinstance(first_choice, dict)
        else {}
    )
    message = choice_payload.get("message", {})
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = choice_payload.get("content", "")
    if not isinstance(content, str):
        raise TypeError("OpenAI-compatible response content must be a string.")
    return _strip_think_blocks(content).strip()


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
    raise ValueError("Structured OpenAI-compatible response did not contain a JSON object.")


def _model_json_schema(schema_model: type[object]) -> dict[str, object]:
    model_json_schema = getattr(schema_model, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    raise TypeError("schema_model must provide model_json_schema().")
