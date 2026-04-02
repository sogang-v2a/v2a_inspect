from __future__ import annotations

import unittest

from v2a_inspect.tools import RemoteGpuPolicy, choose_remote_gpu


class RemoteGpuPolicyTests(unittest.TestCase):
    def test_prefers_a4000_for_16gb_budget(self) -> None:
        selection = choose_remote_gpu(
            RemoteGpuPolicy(
                preferred_sku="A4000",
                fallback_sku="A4500",
                preferred_vram_gb=16,
                max_vram_gb=24,
            )
        )

        self.assertEqual(selection.sku, "A4000")
        self.assertEqual(selection.vram_gb, 16)
        self.assertEqual(selection.source, "preferred")

    def test_falls_back_to_a4500_when_needed(self) -> None:
        selection = choose_remote_gpu(
            RemoteGpuPolicy(
                preferred_sku="A4500",
                fallback_sku="A4000",
                preferred_vram_gb=24,
                max_vram_gb=24,
            )
        )

        self.assertEqual(selection.sku, "A4500")
        self.assertLessEqual(selection.vram_gb, 24)


if __name__ == "__main__":
    unittest.main()
