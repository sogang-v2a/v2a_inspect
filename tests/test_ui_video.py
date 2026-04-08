from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2a_inspect.ui.video import save_uploaded_file


class _FakeUpload:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        return self._payload


class UiVideoTests(unittest.TestCase):
    @patch("v2a_inspect.settings_views.settings")
    def test_save_uploaded_file_uses_shared_dir(self, mock_settings) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_settings.shared_video_dir = Path(tmp_dir)
            saved_path = save_uploaded_file(_FakeUpload("clip.mp4", b"video"))
            self.assertTrue(saved_path.startswith(tmp_dir))
            self.assertEqual(Path(saved_path).read_bytes(), b"video")


if __name__ == "__main__":
    unittest.main()
