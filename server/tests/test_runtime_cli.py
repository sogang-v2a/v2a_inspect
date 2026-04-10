from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v2a_inspect_server.runtime import main


class RuntimeCliTests(unittest.TestCase):
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    def test_check_returns_success_on_available_gpu(self, mock_check) -> None:
        mock_check.return_value = type(
            "CheckResult",
            (),
            {
                "available": True,
                "devices": [],
                "minimum_vram_gb": 10,
                "message": "ok",
            },
        )()
        exit_code = main(["check"])
        self.assertEqual(exit_code, 0)

    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    def test_runtime_info_returns_json(self, mock_server_settings) -> None:
        mock_server_settings.return_value = SimpleNamespace(
            runtime_mode="nvidia_docker",
            runtime_profile="mig10_safe",
            remote_gpu_target="sogang_gpu",
            model_cache_dir=Path(".cache/models"),
            weights_manifest_path=Path("server/model-manifest.json"),
            minimum_gpu_vram_gb=10,
            server_bind_host="0.0.0.0",
            server_bind_port=8080,
            hf_token=None,
        )
        exit_code = main(["runtime-info"])
        self.assertEqual(exit_code, 0)

    @patch("v2a_inspect_server.runtime.WeightsBootstrapper")
    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    def test_bootstrap_uses_manifest(
        self, mock_server_settings, mock_bootstrapper_cls
    ) -> None:
        mock_server_settings.return_value = SimpleNamespace(
            model_cache_dir=Path(".cache/models"),
            weights_manifest_path=Path("server/model-manifest.json"),
            hf_token=None,
        )
        mock_bootstrapper = mock_bootstrapper_cls.return_value
        mock_bootstrapper.load_manifest.return_value = object()
        mock_bootstrapper.ensure_manifest.return_value = {"sam3": Path("/tmp/sam3")}
        exit_code = main(["bootstrap"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
