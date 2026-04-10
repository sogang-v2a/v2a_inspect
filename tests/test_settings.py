from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr

from v2a_inspect.settings import Settings, _resolve_manifest_path
from v2a_inspect.workflows import InspectOptions


class SettingsTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_extra_environment_like_values_are_ignored(self) -> None:
        settings = Settings.model_validate({"openrouter_api_key": SecretStr("secret")})

        self.assertIsNotNone(settings.openrouter_api_key)
        self.assertEqual(Settings.model_config.get("extra"), "ignore")

    def test_mig10_profile_caps_minimum_vram(self) -> None:
        with self.assertRaises(ValueError):
            Settings.model_validate(
                {
                    "runtime_profile": "mig10_safe",
                    "minimum_gpu_vram_gb": 16,
                }
            )

    def test_agentic_bundle_path_is_the_default_research_mode(self) -> None:
        self.assertEqual(Settings().visual_pipeline_mode, "agentic_tool_first")
        self.assertEqual(InspectOptions().pipeline_mode, "agentic_tool_first")

    def test_resolve_manifest_path_falls_back_to_repo_style_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = root / "server" / "model-manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text('{"artifacts": {}}', encoding="utf-8")
            with patch("pathlib.Path.cwd", return_value=root):
                resolved = _resolve_manifest_path(Path("/nonexistent/venv/server/src/v2a_inspect_server/model-manifest.json"))
        self.assertEqual(resolved, manifest)


if __name__ == "__main__":
    unittest.main()
