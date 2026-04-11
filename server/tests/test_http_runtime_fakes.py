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
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    @patch("v2a_inspect_server.runtime._resolve_request_video_path")
    @patch("v2a_inspect_server.runtime._analyze_with_pipeline")
    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_analyze_endpoint_supports_fake_runtime_smoke(
        self,
        mock_build_tooling_runtime,
        mock_analyze_with_pipeline,
        mock_resolve_request_video_path,
        mock_inspect_nvidia_runtime,
    ) -> None:
        mock_inspect_nvidia_runtime.return_value = SimpleNamespace(
            available=True,
            devices=[],
            minimum_vram_gb=10,
            message="ok",
        )
        mock_build_tooling_runtime.return_value = build_fake_tooling_runtime(
            runtime_profile="full_gpu"
        )
        mock_analyze_with_pipeline.return_value = {
            "scene_analysis": SimpleNamespace(
                model_dump=lambda mode="json": {"total_duration": 1.0, "scenes": []}
            ),
            "grouped_analysis": SimpleNamespace(
                model_dump=lambda mode="json": {
                    "scene_analysis": {"total_duration": 1.0, "scenes": []},
                    "raw_tracks": [],
                    "groups": [],
                    "track_to_group": {},
                }
            ),
            "warnings": [],
            "progress_messages": ["done"],
        }
        mock_resolve_request_video_path.return_value = "/tmp/fake.mp4"

        with patch(
            "v2a_inspect_server.runtime.build_final_bundle",
            return_value=SimpleNamespace(
                artifacts=SimpleNamespace(trace_path=None),
                pipeline_metadata={},
                model_dump=lambda mode="json": {
                    "video_id": "video",
                    "video_meta": {
                        "duration_seconds": 1.0,
                        "fps": 2.0,
                        "width": 320,
                        "height": 240,
                    },
                    "candidate_cuts": [],
                    "evidence_windows": [],
                    "physical_sources": [],
                    "sound_events": [],
                    "ambience_beds": [],
                    "generation_groups": [],
                    "validation": {"status": "pass_with_warnings", "issues": []},
                    "artifacts": {},
                    "review_metadata": {
                        "approval_status": "unreviewed",
                        "notes": [],
                        "applied_edits": [],
                    },
                    "pipeline_metadata": {},
                },
            ),
        ):
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
        self.assertIn("multitrack_bundle", payload)


if __name__ == "__main__":
    unittest.main()
