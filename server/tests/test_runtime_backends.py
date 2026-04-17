from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2a_inspect_server.runtime import ToolingRuntime


class ToolingRuntimeBackendTests(unittest.TestCase):
    def test_resident_client_names_only_list_visual_clients(self) -> None:
        runtime = ToolingRuntime(
            bootstrapper=SimpleNamespace(),
            weights_manifest=SimpleNamespace(),
            resolved_artifacts={"sam3": Path("/tmp/sam3"), "embedding": Path("/tmp/emb"), "label": Path("/tmp/label")},
            runtime_profile="cpu_dev",
        )
        runtime._sam3_client = object()
        runtime._embedding_client = object()
        self.assertEqual(runtime.resident_client_names(), ["sam3", "embedding"])

    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    @patch("v2a_inspect_server.runtime.WeightsBootstrapper")
    def test_build_tooling_runtime_reports_missing_bootstrap_artifacts(
        self,
        mock_bootstrapper_cls,
        _mock_gpu,
    ) -> None:
        bootstrapper = mock_bootstrapper_cls.return_value
        bootstrapper.load_manifest.return_value = SimpleNamespace(artifacts=[object()])
        bootstrapper.resolve_manifest.return_value = {
            "sam3": Path("/tmp/missing-sam3"),
            "embedding": Path("/tmp/missing-embedding"),
            "label": Path("/tmp/missing-label"),
        }
        with patch("v2a_inspect_server.runtime.get_server_runtime_settings", return_value=SimpleNamespace(model_cache_dir=Path(".cache/models"), hf_token=None, weights_manifest_path=Path("server/model-manifest.json"), runtime_profile="full_gpu")):
            from v2a_inspect_server.runtime import build_tooling_runtime

            build_tooling_runtime.cache_clear()
            with self.assertRaises(FileNotFoundError):
                build_tooling_runtime()


if __name__ == "__main__":
    unittest.main()
