from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_openai import ChatOpenAI

from v2a_inspect.runtime import build_llm, invoke_structured_llm
from pydantic import BaseModel, Field


class _Ping(BaseModel):
    value: str = Field(min_length=1)


class _StubChatOpenAI(ChatOpenAI):
    def invoke(self, prompt: object, config: object | None = None, **kwargs: object) -> object:
        del prompt, config, kwargs
        return SimpleNamespace(content='<think>reasoning</think>{"value":"ok"}')


class RuntimeLlmTests(unittest.TestCase):
    @patch("langchain_openai.ChatOpenAI", return_value="compat-llm")
    def test_build_llm_prefers_chatopenai_when_base_url_is_set(self, mock_builder) -> None:
        with patch.dict(
            os.environ,
            {
                "V2A_LLM_BASE_URL": "http://127.0.0.1:8080/v1",
                "V2A_LLM_MODEL": "gpt-5.4",
                "V2A_LLM_API_KEY": "",
            },
            clear=False,
        ):
            llm = build_llm(model="gemini-ignored", api_key="unused", timeout_seconds=12.0)
        self.assertEqual(llm, "compat-llm")
        mock_builder.assert_called_once_with(
            model="gpt-5.4",
            base_url="http://127.0.0.1:8080/v1",
            api_key="unused",
            max_retries=3,
            timeout=12.0,
            use_responses_api=False,
        )

    def test_invoke_structured_llm_strips_think_tags_for_chatopenai(self) -> None:
        llm = _StubChatOpenAI(
            model="gpt-5.4",
            base_url="http://127.0.0.1:8080/v1",
            api_key="unused",
            max_retries=1,
            timeout=5,
            use_responses_api=False,
        )
        result = invoke_structured_llm(
            llm=llm,
            schema_model=_Ping,
            prompt="Return {'value':'ok'}",
            method="json_schema",
        )
        self.assertEqual(result.value, "ok")


if __name__ == "__main__":
    unittest.main()
