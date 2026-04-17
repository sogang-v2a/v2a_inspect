from __future__ import annotations

import unittest

from v2a_inspect.openai_compat_llm import _extract_text_content


class OpenAICompatLLMTests(unittest.TestCase):
    def test_extract_text_content_reads_nested_message_content(self) -> None:
        payload = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"value":"ok"}',
                    },
                }
            ]
        }
        self.assertEqual(_extract_text_content(payload), '{"value":"ok"}')


if __name__ == "__main__":
    unittest.main()
