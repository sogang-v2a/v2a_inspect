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
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, RTX A4000, 16384\n",
            stderr="",
        )
        result = inspect_nvidia_runtime(minimum_vram_gb=16)
        self.assertTrue(result.available)
        self.assertEqual(len(result.devices), 1)
