from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from v2a_inspect.clients.server import run_server_inspect
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
    @patch("v2a_inspect.clients.server.request.urlopen")
    def test_run_server_inspect_parses_response(self, mock_urlopen) -> None:
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
            video_path="/data/uploads/video.mp4",
            options=InspectOptions(),
        )
        self.assertEqual(state["warnings"], ["warn"])
        self.assertEqual(state["progress_messages"], ["done"])


if __name__ == "__main__":
    unittest.main()
