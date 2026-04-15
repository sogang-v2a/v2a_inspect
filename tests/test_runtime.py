from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from v2a_inspect.runtime import build_llm


class RuntimeLlmTests(unittest.TestCase):
    @patch("v2a_inspect.runtime.build_openai_compat_llm", return_value="compat-llm")
    def test_build_llm_prefers_openai_compat_when_base_url_is_set(self, mock_builder) -> None:
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
            timeout_seconds=12.0,
        )


if __name__ == "__main__":
    unittest.main()
