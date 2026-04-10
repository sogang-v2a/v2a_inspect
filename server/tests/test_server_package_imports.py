from __future__ import annotations

import subprocess
import unittest


class ServerPackageRootImportTests(unittest.TestCase):
    def test_server_package_root_import_stays_lightweight(self) -> None:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                "server",
                "python",
                "-c",
                (
                    "import sys, v2a_inspect_server; "
                    "assert 'transformers' not in sys.modules; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)

    def test_ui_package_root_import_stays_lightweight(self) -> None:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                (
                    "import sys, v2a_inspect.ui; "
                    "assert 'streamlit' not in sys.modules"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
