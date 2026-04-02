from __future__ import annotations

import unittest
from pathlib import Path
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
                "minimum_vram_gb": 16,
                "message": "ok",
            },
        )()
        exit_code = main(["check"])
        self.assertEqual(exit_code, 0)

    @patch("v2a_inspect_server.runtime.settings")
    def test_runtime_info_returns_json(self, mock_settings) -> None:
        mock_settings.runtime_mode = "nvidia_docker"
        mock_settings.model_cache_dir = Path(".cache/models")
        mock_settings.weights_manifest_path = Path("server/model-manifest.json")
        mock_settings.minimum_gpu_vram_gb = 16
        mock_settings.server_bind_host = "0.0.0.0"
        mock_settings.server_bind_port = 8080
        exit_code = main(["runtime-info"])
        self.assertEqual(exit_code, 0)

    @patch("v2a_inspect_server.runtime.WeightsBootstrapper")
    @patch("v2a_inspect_server.runtime.settings")
    def test_bootstrap_uses_manifest(
        self, mock_settings, mock_bootstrapper_cls
    ) -> None:
        mock_settings.model_cache_dir = Path(".cache/models")
        mock_settings.weights_manifest_path = Path("server/model-manifest.json")
        mock_settings.hf_token = None
        mock_bootstrapper = mock_bootstrapper_cls.return_value
        mock_bootstrapper.load_manifest.return_value = object()
        mock_bootstrapper.ensure_manifest.return_value = {"sam3": Path("/tmp/sam3")}
        exit_code = main(["bootstrap"])
        self.assertEqual(exit_code, 0)
