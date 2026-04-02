from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2a_inspect_server.runtime import _resolve_request_video_path


class RequestVideoPathTests(unittest.TestCase):
    def test_reuses_existing_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"video")
            resolved = _resolve_request_video_path(
                video_path=str(video_path),
                video_filename="clip.mp4",
                video_base64=None,
            )
        self.assertEqual(resolved, str(video_path))

    def test_decodes_base64_when_path_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("v2a_inspect.settings.settings.shared_video_dir", Path(tmp_dir)):
                resolved = _resolve_request_video_path(
                    video_path="",
                    video_filename="clip.mp4",
                    video_base64=base64.b64encode(b"video").decode("ascii"),
                )
            self.assertEqual(Path(resolved).read_bytes(), b"video")


if __name__ == "__main__":
    unittest.main()
