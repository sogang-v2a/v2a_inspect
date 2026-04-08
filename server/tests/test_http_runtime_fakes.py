from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib import request
from unittest.mock import patch

from server.tests.fakes import build_fake_tooling_runtime
from v2a_inspect_server.runtime import _build_handler


class RuntimeHttpFakeSmokeTests(unittest.TestCase):
    @patch("v2a_inspect_server.runtime.get_grouped_analysis")
    @patch("v2a_inspect_server.runtime._resolve_request_video_path")
    @patch("v2a_inspect_server.runtime.run_inspect")
    @patch("v2a_inspect_server.runtime.build_tool_context")
    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_analyze_endpoint_supports_fake_runtime_smoke(
        self,
        mock_build_tooling_runtime,
        mock_build_tool_context,
        mock_run_inspect,
        mock_resolve_request_video_path,
        mock_get_grouped_analysis,
    ) -> None:
        mock_build_tooling_runtime.return_value = build_fake_tooling_runtime()
        mock_build_tool_context.return_value = {
            "progress_messages": ["fake-tool-step"],
            "tool_grouping_hints": "fake grouping hints",
        }
        mock_run_inspect.return_value = {
            "scene_analysis": SimpleNamespace(
                model_dump=lambda mode="json": {"total_duration": 1.0, "scenes": []}
            ),
            "warnings": [],
            "progress_messages": ["done"],
        }
        mock_resolve_request_video_path.return_value = "/tmp/fake.mp4"
        mock_get_grouped_analysis.return_value = SimpleNamespace(
            model_dump=lambda mode="json": {
                "scene_analysis": {"total_duration": 1.0, "scenes": []},
                "raw_tracks": [],
                "groups": [],
                "track_to_group": {},
            }
        )

        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            response = request.urlopen(
                request.Request(
                    f"http://127.0.0.1:{server.server_port}/analyze",
                    data=json.dumps(
                        {
                            "video_path": "/tmp/fake.mp4",
                            "video_filename": "fake.mp4",
                            "options": {},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
            ).read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(payload["progress_messages"], ["done"])


if __name__ == "__main__":
    unittest.main()
