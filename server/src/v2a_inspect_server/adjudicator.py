from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class IssueAdjudication(BaseModel):
    resolution: Literal["accept", "run_tool"] = "run_tool"
    tool_name: Literal[
        "refine_candidate_cuts",
        "densify_window_sampling",
        "propose_source_hypotheses",
        "verify_scene_hypotheses",
        "build_source_semantics",
        "rerun_description_writer",
        "validate_bundle",
    ] | None = None
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class GeminiIssueJudge:
    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str,
        max_retries: int = 1,
        timeout_seconds: float = 30.0,
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

    def judge_issue(self, context: Mapping[str, object]) -> IssueAdjudication | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = [
            SystemMessage(
                content=(
                    "You adjudicate ambiguous video-understanding repair decisions. "
                    "Use only the provided structured evidence. Prefer keep_current "
                    "when the issue looks benign. Choose a tool only when it is the "
                    "smallest action that could realistically improve the bundle."
                )
            ),
            HumanMessage(
                content=(
                    "Return one typed adjudication for this issue.\n"
                    "Use resolution=accept when the current bundle should stand.\n"
                    "Use resolution=run_tool and set tool_name when a repair action is warranted.\n\n"
                    f"Structured evidence:\n{json.dumps(dict(context), indent=2, ensure_ascii=False, default=str)}"
                )
            ),
        ]
        structured_llm = self.llm.with_structured_output(
            IssueAdjudication,
            method="json_schema",
        )
        result = structured_llm.invoke(prompt)
        if isinstance(result, IssueAdjudication):
            return result
        return IssueAdjudication.model_validate(result)
