from __future__ import annotations

import unittest

from v2a_inspect.pipeline.prompt_templates import ResolvedPrompt
from v2a_inspect.pipeline.nodes._shared import append_prompt_evidence


class PromptEvidenceTests(unittest.TestCase):
    def test_append_prompt_evidence_adds_block(self) -> None:
        prompt = ResolvedPrompt(
            name="grouping",
            system_text="system",
            user_text="hello",
            source="local",
        )
        updated = append_prompt_evidence(
            prompt,
            title="Tool grouping hints",
            evidence="cluster g0: t0,t1",
        )
        self.assertIn("[Tool grouping hints]", updated.user_text)
        self.assertIn("cluster g0: t0,t1", updated.user_text)


if __name__ == "__main__":
    unittest.main()
