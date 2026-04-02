from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2a_inspect.clients.server import _build_request_payload, run_server_inspect
from v2a_inspect.workflows import InspectOptions


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ServerClientTests(unittest.TestCase):
    def test_build_request_payload_includes_inline_video_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"video-data")
            payload = _build_request_payload(
                video_path=str(video_path),
                options=InspectOptions(),
            )
        self.assertEqual(payload["video_filename"], "clip.mp4")
        self.assertEqual(
            payload["video_base64"],
            base64.b64encode(b"video-data").decode("ascii"),
        )

    @patch("v2a_inspect.clients.server.request.urlopen")
    def test_run_server_inspect_parses_response(self, mock_urlopen) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"video-data")
            mock_urlopen.return_value = _FakeResponse(
                {
                    "scene_analysis": {
                        "total_duration": 1.0,
                        "scenes": [],
                    },
                    "grouped_analysis": {
                        "scene_analysis": {
                            "total_duration": 1.0,
                            "scenes": [],
                        },
                        "raw_tracks": [],
                        "groups": [],
                        "track_to_group": {},
                    },
                    "warnings": ["warn"],
                    "progress_messages": ["done"],
                }
            )
            state = run_server_inspect(
                server_base_url="http://server:8080",
                video_path=str(video_path),
                options=InspectOptions(),
            )
        self.assertEqual(state["warnings"], ["warn"])
        self.assertEqual(state["progress_messages"], ["done"])


if __name__ == "__main__":
    unittest.main()
