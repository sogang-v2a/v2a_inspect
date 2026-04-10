from __future__ import annotations

import unittest

from v2a_inspect.tools import RemoteGpuPolicy, choose_remote_gpu


class RemoteGpuPolicyTests(unittest.TestCase):
    def test_prefers_mig10_profile_for_university_gpu(self) -> None:
        selection = choose_remote_gpu(
            RemoteGpuPolicy(
                target="sogang_gpu",
                preferred_profile="mig10_safe",
                fallback_profile="full_gpu",
                preferred_vram_gb=10,
                max_vram_gb=80,
            )
        )

        self.assertEqual(selection.target, "sogang_gpu")
        self.assertEqual(selection.runtime_profile, "mig10_safe")
        self.assertEqual(selection.vram_gb, 10)
        self.assertEqual(selection.source, "preferred")

    def test_falls_back_to_full_gpu_when_requested_budget_is_larger(self) -> None:
        selection = choose_remote_gpu(
            RemoteGpuPolicy(
                target="sogang_gpu",
                preferred_profile="mig10_safe",
                fallback_profile="full_gpu",
                preferred_vram_gb=20,
                max_vram_gb=40,
            )
        )

        self.assertEqual(selection.runtime_profile, "full_gpu")
        self.assertEqual(selection.vram_gb, 20)
        self.assertEqual(selection.source, "fallback")


if __name__ == "__main__":
    unittest.main()
