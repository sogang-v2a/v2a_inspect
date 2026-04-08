from __future__ import annotations

import subprocess
import unittest


class PackageRootImportTests(unittest.TestCase):
    def test_client_package_root_import_stays_lightweight(self) -> None:
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-c",
                (
                    "import sys, v2a_inspect; "
                    "assert 'google.genai' not in sys.modules; "
                    "assert 'langchain_google_genai' not in sys.modules"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
