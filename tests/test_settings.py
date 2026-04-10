from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import SecretStr

from v2a_inspect.settings import Settings


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


if __name__ == "__main__":
    unittest.main()
