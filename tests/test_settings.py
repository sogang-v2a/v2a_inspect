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

    def test_gpu_policy_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            Settings.model_validate(
                {
                    "remote_gpu_preference": "A4000",
                    "remote_gpu_fallback": "A4000",
                }
            )


if __name__ == "__main__":
    unittest.main()
