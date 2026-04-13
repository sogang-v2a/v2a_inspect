from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2a_inspect_server.runtime import _resolve_request_video_path


class RequestVideoPathTests(unittest.TestCase):
    def test_reuses_existing_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_root = Path(tmp_dir) / "uploads"
            upload_root.mkdir(parents=True, exist_ok=True)
            video_path = upload_root / "clip.mp4"
            video_path.write_bytes(b"video")
            from unittest.mock import patch

            with patch(
                "v2a_inspect_server.settings.settings.shared_video_dir", upload_root
            ):
                resolved = _resolve_request_video_path(
                    video_path=str(video_path),
                    video_filename="clip.mp4",
                )
        self.assertEqual(resolved, str(video_path))

    def test_rejects_paths_outside_managed_upload_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_root = Path(tmp_dir) / "uploads"
            upload_root.mkdir(parents=True, exist_ok=True)
            video_path = Path(tmp_dir) / "other.mp4"
            video_path.write_bytes(b"video")
            from unittest.mock import patch

            with patch(
                "v2a_inspect_server.settings.settings.shared_video_dir", upload_root
            ):
                with self.assertRaises(ValueError):
                    _resolve_request_video_path(
                        video_path=str(video_path),
                        video_filename="other.mp4",
                    )

    def test_rejects_missing_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_root = Path(tmp_dir) / "uploads"
            upload_root.mkdir(parents=True, exist_ok=True)
            from unittest.mock import patch

            with patch(
                "v2a_inspect_server.settings.settings.shared_video_dir", upload_root
            ):
                with self.assertRaises(ValueError):
                    _resolve_request_video_path(
                        video_path="",
                        video_filename="clip.mp4",
                    )

    def test_rejects_missing_file_even_under_upload_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            upload_root = Path(tmp_dir) / "uploads"
            upload_root.mkdir(parents=True, exist_ok=True)
            from unittest.mock import patch

            with patch(
                "v2a_inspect_server.settings.settings.shared_video_dir", upload_root
            ):
                with self.assertRaises(ValueError):
                    _resolve_request_video_path(
                        video_path=str(upload_root / "missing.mp4"),
                        video_filename="clip.mp4",
                    )


if __name__ == "__main__":
    unittest.main()
