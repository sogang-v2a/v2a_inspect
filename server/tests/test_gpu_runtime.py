from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from v2a_inspect_server.gpu_runtime import inspect_nvidia_runtime


class GpuRuntimeTests(unittest.TestCase):
    @patch("v2a_inspect_server.gpu_runtime.shutil.which", return_value=None)
    def test_missing_nvidia_smi_fails_cleanly(self, _mock_which) -> None:
        result = inspect_nvidia_runtime(minimum_vram_gb=16)
        self.assertFalse(result.available)
        self.assertEqual(result.devices, [])

    @patch(
        "v2a_inspect_server.gpu_runtime.shutil.which",
        return_value="/usr/bin/nvidia-smi",
    )
    @patch("v2a_inspect_server.gpu_runtime.subprocess.run")
    def test_detects_eligible_gpu(self, mock_run, _mock_which) -> None:
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="0, RTX A4000, 16384\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="GPU 0: RTX A4000 (UUID: gpu-0)\n",
                stderr="",
            ),
        ]
        result = inspect_nvidia_runtime(minimum_vram_gb=16)
        self.assertTrue(result.available)
        self.assertEqual(len(result.devices), 1)

    @patch(
        "v2a_inspect_server.gpu_runtime.shutil.which",
        return_value="/usr/bin/nvidia-smi",
    )
    @patch("v2a_inspect_server.gpu_runtime.subprocess.run")
    def test_uses_mig_inventory_when_memory_query_is_blocked(
        self, mock_run, _mock_which
    ) -> None:
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="0, NVIDIA A100 80GB PCIe, [Insufficient Permissions]\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-0)\n"
                    "  MIG 1g.10gb     Device  0: (UUID: MIG-0)\n"
                ),
                stderr="",
            ),
        ]
        result = inspect_nvidia_runtime(minimum_vram_gb=10)
        self.assertTrue(result.available)
        self.assertEqual(result.devices[0].memory_total_mib, 10240)
