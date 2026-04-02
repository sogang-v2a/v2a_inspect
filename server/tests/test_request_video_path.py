from __future__ import annotations

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
                video_url=None,
            )
        self.assertEqual(resolved, str(video_path))

    def test_downloads_video_url_when_path_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("v2a_inspect.settings.settings.shared_video_dir", Path(tmp_dir)):
                with patch("urllib.request.urlretrieve") as mock_urlretrieve:
                    def _fake_urlretrieve(url: str, target: Path) -> None:
                        Path(target).write_bytes(b"video")

                    mock_urlretrieve.side_effect = _fake_urlretrieve
                    resolved = _resolve_request_video_path(
                        video_path="",
                        video_filename="clip.mp4",
                        video_url="https://example.com/video.mp4",
                    )
            self.assertEqual(Path(resolved).read_bytes(), b"video")
            self.assertTrue(Path(resolved).name.endswith(".mp4"))


if __name__ == "__main__":
    unittest.main()
