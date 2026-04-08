from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import SecretStr

from v2a_inspect.settings import Settings
from v2a_inspect.settings_views import (
    get_client_runtime_settings,
    get_server_runtime_settings,
    required_env_vars_by_runtime_mode,
)


class SettingsViewTests(unittest.TestCase):
    def test_client_and_server_views_split_runtime_concerns(self) -> None:
        base = Settings.model_construct(
            server_base_url="https://server.example",
            shared_video_dir=Path("/data/uploads"),
            remote_timeout_seconds=120,
            ui_analysis_concurrency_limit=2,
            ui_analysis_acquire_timeout_seconds=120,
            ui_temp_cleanup_max_age_seconds=3600,
            ui_cleanup_interval_seconds=1800,
            runtime_mode="nvidia_docker",
            server_bind_host="0.0.0.0",
            server_bind_port=8080,
            minimum_gpu_vram_gb=16,
            model_cache_dir=Path("/data/models"),
            weights_manifest_path=Path("server/model-manifest.json"),
            hf_token=SecretStr("hf-secret"),
        )

        client_view = get_client_runtime_settings(base)
        server_view = get_server_runtime_settings(base)

        self.assertEqual(client_view.server_base_url, "https://server.example")
        self.assertEqual(server_view.model_cache_dir, Path("/data/models"))
        self.assertEqual(server_view.hf_token, "hf-secret")
        self.assertFalse(hasattr(client_view, "model_cache_dir"))

    def test_required_env_var_reference_is_stable(self) -> None:
        envs = required_env_vars_by_runtime_mode()
        self.assertIn("client_ui", envs)
        self.assertIn("server_runtime", envs)
        self.assertIn("HF_TOKEN", envs["server_runtime"])


if __name__ == "__main__":
    unittest.main()
