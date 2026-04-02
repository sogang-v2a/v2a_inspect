from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib import request
from unittest.mock import patch

from v2a_inspect_server.runtime import _build_handler


class RuntimeHttpTests(unittest.TestCase):
    @patch("v2a_inspect_server.runtime.settings")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    def test_health_endpoint_returns_json(self, mock_check, mock_settings) -> None:
        mock_settings.runtime_mode = "nvidia_docker"
        mock_settings.minimum_gpu_vram_gb = 16
        mock_check.return_value = SimpleNamespace(
            available=True,
            devices=[],
            minimum_vram_gb=16,
            message="ok",
        )

        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            response = request.urlopen(
                f"http://127.0.0.1:{server.server_port}/health"
            ).read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["runtime_mode"], "nvidia_docker")

    @patch("v2a_inspect_server.runtime.settings")
    @patch("v2a_inspect_server.runtime.get_grouped_analysis")
    @patch("v2a_inspect_server.runtime.run_inspect")
    @patch("v2a_inspect_server.runtime.build_tool_context")
    def test_analyze_endpoint_returns_grouped_payload(
        self,
        mock_build_tool_context,
        mock_run_inspect,
        mock_get_grouped_analysis,
        mock_settings,
    ) -> None:
        mock_settings.runtime_mode = "nvidia_docker"
        mock_settings.minimum_gpu_vram_gb = 16
        mock_build_tool_context.return_value = {
            "progress_messages": ["tool-step"],
        }

        mock_run_inspect.return_value = {
            "scene_analysis": SimpleNamespace(
                model_dump=lambda mode="json": {"total_duration": 1.0, "scenes": []}
            ),
            "warnings": ["warn"],
            "progress_messages": ["done"],
        }
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
                            "video_path": "/data/uploads/video.mp4",
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
        self.assertEqual(payload["warnings"], ["warn"])
        self.assertEqual(payload["progress_messages"], ["done"])


if __name__ == "__main__":
    unittest.main()
