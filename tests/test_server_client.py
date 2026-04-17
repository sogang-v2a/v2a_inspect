from __future__ import annotations

import unittest
from unittest.mock import patch

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.clients.server import run_server_inspect, run_server_inspect_raw
from v2a_inspect.workflows import InspectOptions


class ServerClientTests(unittest.TestCase):
    @patch("v2a_inspect.clients.server.run_local_inspect_raw")
    def test_run_server_inspect_parses_local_orchestrator_response(self, mock_run_local_inspect_raw) -> None:
        mock_run_local_inspect_raw.return_value = {
            "multitrack_bundle": {
                "video_id": "clip",
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
                "review_metadata": {"approval_status": "unreviewed", "notes": [], "applied_edits": []},
                "pipeline_metadata": {},
            },
            "warnings": ["warn"],
            "progress_messages": ["done"],
        }
        state = run_server_inspect(
            server_base_url="http://server:8080",
            video_path="/tmp/clip.mp4",
            options=InspectOptions(),
        )
        self.assertEqual(state["warnings"], ["warn"])
        self.assertEqual(state["progress_messages"], ["done"])
        self.assertIsInstance(state.get("multitrack_bundle"), MultitrackDescriptionBundle)

    @patch("v2a_inspect.clients.server.run_local_inspect_raw")
    def test_run_server_inspect_raw_preserves_server_base_url(self, mock_run_local_inspect_raw) -> None:
        mock_run_local_inspect_raw.return_value = {
            "multitrack_bundle": {
                "video_id": "clip",
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
                "review_metadata": {"approval_status": "unreviewed", "notes": [], "applied_edits": []},
                "pipeline_metadata": {},
            },
            "effective_runtime_profile": "full_gpu",
        }
        payload = run_server_inspect_raw(
            server_base_url="http://server:8080",
            video_path="/tmp/clip.mp4",
            options=InspectOptions(),
        )
        self.assertEqual(payload["effective_runtime_profile"], "full_gpu")
        passed_options = mock_run_local_inspect_raw.call_args.kwargs["options"]
        self.assertEqual(passed_options.server_base_url, "http://server:8080")


if __name__ == "__main__":
    unittest.main()
