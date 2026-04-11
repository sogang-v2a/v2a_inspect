from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.clients.server import (
    _build_request_payload,
    _upload_video,
    run_server_inspect,
    run_server_inspect_raw,
)
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
    def test_build_request_payload_uses_remote_video_path(self) -> None:
        payload = _build_request_payload(
            video_path="/tmp/clip.mp4",
            remote_video_path="/server/uploads/clip.mp4",
            options=InspectOptions(),
        )
        self.assertEqual(payload["video_filename"], "clip.mp4")
        self.assertEqual(payload["video_path"], "/server/uploads/clip.mp4")

    @patch("v2a_inspect.clients.server.request.urlopen")
    def test_upload_video_returns_remote_path(self, mock_urlopen) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"video-data")
            mock_urlopen.return_value = _FakeResponse(
                {
                    "ok": True,
                    "video_path": "/remote/tmp/clip.mp4",
                }
            )
            remote_path = _upload_video(
                server_base_url="http://server:8080",
                video_path=str(video_path),
                timeout_seconds=30,
            )
        self.assertEqual(remote_path, "/remote/tmp/clip.mp4")

    @patch("v2a_inspect.clients.server.request.urlopen")
    def test_run_server_inspect_parses_response(self, mock_urlopen) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"video-data")
            mock_urlopen.side_effect = [
                _FakeResponse({"ok": True, "video_path": "/remote/tmp/clip.mp4"}),
                _FakeResponse(
                    {
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
                ),
            ]
            state = run_server_inspect(
                server_base_url="http://server:8080",
                video_path=str(video_path),
                options=InspectOptions(),
            )
        self.assertEqual(state["warnings"], ["warn"])
        self.assertEqual(state["progress_messages"], ["done"])
        self.assertIsInstance(state.get("multitrack_bundle"), MultitrackDescriptionBundle)

    @patch("v2a_inspect.clients.server.request.urlopen")
    def test_run_server_inspect_raw_preserves_runtime_metadata(self, mock_urlopen) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "clip.mp4"
            video_path.write_bytes(b"video-data")
            mock_urlopen.side_effect = [
                _FakeResponse({"ok": True, "video_path": "/remote/tmp/clip.mp4"}),
                _FakeResponse(
                    {
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
                        "warnings": [],
                        "progress_messages": [],
                        "effective_runtime_profile": "full_gpu",
                        "warm_start": True,
                    }
                ),
            ]
            payload = run_server_inspect_raw(
                server_base_url="http://server:8080",
                video_path=str(video_path),
                options=InspectOptions(),
            )
        self.assertEqual(payload["effective_runtime_profile"], "full_gpu")
        self.assertTrue(payload["warm_start"])


if __name__ == "__main__":
    unittest.main()
