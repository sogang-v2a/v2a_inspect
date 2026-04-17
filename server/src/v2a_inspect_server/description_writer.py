from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.runtime import invoke_structured_llm

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class DescriptionDraft(BaseModel):
    canonical_description: str = Field(min_length=1)
    description_confidence: float = Field(ge=0.0, le=1.0)
    description_rationale: str = Field(min_length=1)


class DescriptionWriterLike(Protocol):
    def write_group_description(
        self, context: Mapping[str, object]
    ) -> DescriptionDraft | None: ...


class GeminiDescriptionWriter:
    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str,
        max_retries: int = 1,
        timeout_seconds: float = 45.0,
    ) -> None:
        self._llm: BaseChatModel | None = None
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            from v2a_inspect.runtime import build_llm

            self._llm = build_llm(
                model=self._model,
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout_seconds=self._timeout_seconds,
            )
        return self._llm

    def write_group_description(
        self, context: Mapping[str, object]
    ) -> DescriptionDraft | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = [
            SystemMessage(
                content=(
                    "You write concise canonical audio descriptions for downstream "
                    "audio generation. Ground every description in the provided "
                    "structured evidence only. Do not invent invisible causes, "
                    "extra sound layers, or unsupported identities."
                )
            ),
            HumanMessage(
                content=(
                    "Write one canonical audio description for this generation group.\n"
                    "Return compact generation-ready wording, a calibrated confidence, "
                    "and a short rationale.\n\n"
                    f"Structured evidence:\n{json.dumps(dict(context), indent=2, ensure_ascii=False)}"
                )
            ),
        ]
        return invoke_structured_llm(
            llm=self.llm,
            schema_model=DescriptionDraft,
            prompt=prompt,
            method="json_schema",
        )
