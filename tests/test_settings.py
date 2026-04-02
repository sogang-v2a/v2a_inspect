from __future__ import annotations

import unittest

from pydantic import SecretStr

from v2a_inspect.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_extra_environment_like_values_are_ignored(self) -> None:
        settings = Settings(
            runpod_api_key=SecretStr("runpod-secret"),
        )

        self.assertEqual(
            settings.runpod_api_key.get_secret_value()
            if settings.runpod_api_key
            else None,
            "runpod-secret",
        )
        self.assertEqual(Settings.model_config.get("extra"), "ignore")

    def test_gpu_policy_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            Settings(
                remote_gpu_preference="A4000",
                remote_gpu_fallback="A4000",
            )


if __name__ == "__main__":
    unittest.main()
