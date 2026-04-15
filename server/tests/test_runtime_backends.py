from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2a_inspect_server.runtime import ToolingRuntime


class ToolingRuntimeBackendTests(unittest.TestCase):
    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    @patch("v2a_inspect_server.runtime.WeightsBootstrapper")
    @patch("v2a_inspect_server.runtime.WeightsManifest")
    @patch("v2a_inspect_server.gemini_source_proposal.GeminiSourceProposer", return_value="proposer")
    def test_source_proposer_is_available_with_openai_compat_backend(
        self,
        mock_proposer,
        _mock_manifest,
        _mock_bootstrapper,
        _mock_gpu,
        mock_settings,
    ) -> None:
        mock_settings.return_value = SimpleNamespace(gemini_api_key=None)
        runtime = ToolingRuntime(
            bootstrapper=SimpleNamespace(),
            weights_manifest=SimpleNamespace(),
            resolved_artifacts={"sam3": Path("/tmp/sam3"), "embedding": Path("/tmp/emb"), "label": Path("/tmp/label")},
            runtime_profile="cpu_dev",
        )
        with patch.dict(os.environ, {"V2A_LLM_BASE_URL": "http://127.0.0.1:8080/v1"}, clear=False):
            proposer = runtime.source_proposer
        self.assertEqual(proposer, "proposer")
        mock_proposer.assert_called_once_with(api_key="")

    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    @patch("v2a_inspect_server.description_writer.GeminiDescriptionWriter", return_value="writer")
    def test_description_writer_accepts_string_server_key_without_secret_wrapper(
        self,
        mock_writer,
        mock_settings,
    ) -> None:
        mock_settings.return_value = SimpleNamespace(gemini_api_key="plain-text-key")
        runtime = ToolingRuntime(
            bootstrapper=SimpleNamespace(),
            weights_manifest=SimpleNamespace(),
            resolved_artifacts={"sam3": Path("/tmp/sam3"), "embedding": Path("/tmp/emb"), "label": Path("/tmp/label")},
            runtime_profile="cpu_dev",
        )
        writer = runtime.description_writer
        self.assertEqual(writer, "writer")
        mock_writer.assert_called_once_with(api_key="plain-text-key")


if __name__ == "__main__":
    unittest.main()
