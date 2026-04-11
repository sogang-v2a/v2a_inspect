from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib import request
from unittest.mock import patch

from v2a_inspect_server.runtime import _build_handler


class RuntimeHttpTests(unittest.TestCase):
    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    def test_healthz_endpoint_returns_lightweight_json(self, mock_check, mock_build_tooling_runtime, mock_server_settings) -> None:
        mock_server_settings.return_value = SimpleNamespace(
            runtime_mode="nvidia_docker",
            runtime_profile="full_gpu",
            remote_gpu_target="sogang_gpu",
            minimum_gpu_vram_gb=10,
            model_cache_dir=Path('/tmp/models'),
            weights_manifest_path=Path('/tmp/model-manifest.json'),
            server_bind_host='127.0.0.1',
            server_bind_port=8080,
            shared_video_dir=Path('/tmp'),
            hf_token=None,
        )
        mock_build_tooling_runtime.return_value = SimpleNamespace()
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
                f"http://127.0.0.1:{server.server_port}/healthz"
            ).read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["runtime_mode"], "nvidia_docker")
        self.assertEqual(payload["runtime_profile"], "full_gpu")
        mock_build_tooling_runtime.assert_not_called()
        mock_check.assert_not_called()

    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    def test_readyz_endpoint_returns_runtime_readiness(self, mock_check, mock_build_tooling_runtime, mock_server_settings) -> None:
        mock_server_settings.return_value = SimpleNamespace(
            runtime_mode="nvidia_docker",
            runtime_profile="full_gpu",
            remote_gpu_target="sogang_gpu",
            minimum_gpu_vram_gb=10,
            model_cache_dir=Path('/tmp/models'),
            weights_manifest_path=Path('/tmp/model-manifest.json'),
            server_bind_host='127.0.0.1',
            server_bind_port=8080,
            shared_video_dir=Path('/tmp'),
            hf_token=None,
        )
        mock_build_tooling_runtime.return_value = SimpleNamespace(
            artifacts_missing=lambda: [],
            resident_client_names=lambda: [],
            residency_mode="resident",
        )
        mock_check.return_value = SimpleNamespace(
            available=True,
            devices=[],
            minimum_vram_gb=10,
            message="ok",
        )

        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            response = request.urlopen(
                f"http://127.0.0.1:{server.server_port}/readyz"
            ).read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["bootstrap_ready"])
        self.assertEqual(payload["runtime_profile"], "full_gpu")
        self.assertEqual(payload["residency_mode"], "resident")

    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    def test_warmup_endpoint_loads_models_and_keeps_them_resident(self, mock_check, mock_build_tooling_runtime, mock_server_settings) -> None:
        mock_server_settings.return_value = SimpleNamespace(
            runtime_mode="nvidia_docker",
            runtime_profile="full_gpu",
            remote_gpu_target="sogang_gpu",
            minimum_gpu_vram_gb=10,
            model_cache_dir=Path('/tmp/models'),
            weights_manifest_path=Path('/tmp/model-manifest.json'),
            server_bind_host='127.0.0.1',
            server_bind_port=8080,
            shared_video_dir=Path('/tmp'),
            hf_token=None,
        )
        mock_build_tooling_runtime.return_value = SimpleNamespace(
            artifacts_missing=lambda: [],
            warmup_visual_clients=lambda: {
                "model_load_status": {"sam3": "ready", "embedding": "ready", "label": "ready"},
                "model_load_seconds": {"sam3": 1.0, "embedding": 0.5, "label": 0.25},
                "resident_models": ["sam3", "embedding", "label"],
            },
            resident_client_names=lambda: ["sam3", "embedding", "label"],
            residency_mode="resident",
        )
        mock_check.return_value = SimpleNamespace(
            available=True,
            devices=[],
            minimum_vram_gb=10,
            message="ok",
        )

        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            response = request.urlopen(
                request.Request(
                    f"http://127.0.0.1:{server.server_port}/warmup",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
            ).read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertTrue(payload["warmup_ok"])
        self.assertEqual(payload["resident_models"], ["sam3", "embedding", "label"])
        self.assertEqual(payload["residency_mode"], "resident")

    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    @patch("v2a_inspect_server.runtime._resolve_request_video_path")
    @patch("v2a_inspect_server.runtime._analyze_with_pipeline")
    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_analyze_endpoint_returns_grouped_payload(
        self,
        mock_build_tooling_runtime,
        mock_analyze_with_pipeline,
        mock_resolve_request_video_path,
        mock_inspect_nvidia_runtime,
        mock_server_settings,
    ) -> None:
        mock_server_settings.return_value = SimpleNamespace(
            runtime_mode="nvidia_docker",
            runtime_profile="full_gpu",
            remote_gpu_target="sogang_gpu",
            minimum_gpu_vram_gb=10,
            model_cache_dir=Path('.cache/models'),
            weights_manifest_path=Path('server/model-manifest.json'),
            server_bind_host='127.0.0.1',
            server_bind_port=8080,
            shared_video_dir=Path('/tmp'),
            hf_token=None,
        )
        mock_build_tooling_runtime.return_value = SimpleNamespace(
            runtime_profile="full_gpu",
            residency_mode="resident",
            resident_client_names=lambda: ["sam3", "embedding", "label"],
        )
        mock_inspect_nvidia_runtime.return_value = SimpleNamespace(
            available=True,
            devices=[],
            minimum_vram_gb=10,
            message="ok",
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
            "warnings": ["warn"],
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
                    "review_metadata": {"approval_status": "unreviewed", "notes": [], "applied_edits": []},
                    "pipeline_metadata": {},
                }
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
                                "video_path": "/data/uploads/video.mp4",
                                "video_filename": "video.mp4",
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
        self.assertIn("multitrack_bundle", payload)
        self.assertEqual(payload["effective_runtime_profile"], "full_gpu")
        self.assertEqual(payload["runtime_profile_source"], "server_settings")
        self.assertEqual(payload["residency_mode"], "resident")

    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    @patch("v2a_inspect_server.runtime._resolve_request_video_path")
    @patch("v2a_inspect_server.runtime._analyze_with_pipeline")
    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_analyze_endpoint_skips_review_pass_for_agentic_mode(
        self,
        mock_build_tooling_runtime,
        mock_analyze_with_pipeline,
        mock_resolve_request_video_path,
        mock_inspect_nvidia_runtime,
        mock_server_settings,
    ) -> None:
        mock_server_settings.return_value = SimpleNamespace(
            runtime_mode="nvidia_docker",
            runtime_profile="full_gpu",
            remote_gpu_target="sogang_gpu",
            minimum_gpu_vram_gb=10,
            model_cache_dir=Path('.cache/models'),
            weights_manifest_path=Path('server/model-manifest.json'),
            server_bind_host='127.0.0.1',
            server_bind_port=8080,
            shared_video_dir=Path('/tmp'),
            hf_token=None,
        )
        mock_build_tooling_runtime.return_value = SimpleNamespace(
            runtime_profile="full_gpu",
            residency_mode="resident",
            resident_client_names=lambda: ["sam3", "embedding", "label"],
        )
        mock_inspect_nvidia_runtime.return_value = SimpleNamespace(
            available=True,
            devices=[],
            minimum_vram_gb=10,
            message="ok",
        )
        bundle = SimpleNamespace(
            artifacts=SimpleNamespace(trace_path=None),
            pipeline_metadata={},
            model_dump=lambda mode="json": {"video_id": "video"},
        )
        grouped = SimpleNamespace(
            model_dump=lambda mode="json": {
                "scene_analysis": {"total_duration": 1.0, "scenes": []},
                "raw_tracks": [],
                "groups": [],
                "track_to_group": {},
            }
        )
        mock_analyze_with_pipeline.return_value = {
            "scene_analysis": SimpleNamespace(
                model_dump=lambda mode="json": {"total_duration": 1.0, "scenes": []}
            ),
            "grouped_analysis": grouped,
            "multitrack_bundle": bundle,
            "warnings": [],
            "progress_messages": [],
        }
        mock_resolve_request_video_path.return_value = "/tmp/fake.mp4"

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
                            "video_filename": "video.mp4",
                            "options": {"pipeline_mode": "agentic_tool_first"},
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
        self.assertIn("multitrack_bundle", payload)

    @patch("v2a_inspect_server.runtime._resolve_request_video_path")
    @patch("v2a_inspect_server.runtime.inspect_nvidia_runtime")
    def test_analyze_endpoint_rejects_when_gpu_runtime_is_unavailable(
        self,
        mock_inspect_nvidia_runtime,
        mock_resolve_request_video_path,
    ) -> None:
        mock_resolve_request_video_path.return_value = "/tmp/fake.mp4"
        mock_inspect_nvidia_runtime.return_value = SimpleNamespace(
            available=False,
            devices=[],
            minimum_vram_gb=10,
            message="missing gpu",
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            with self.assertRaises(request.HTTPError) as raised:
                request.urlopen(
                    request.Request(
                        f"http://127.0.0.1:{server.server_port}/analyze",
                        data=json.dumps(
                            {
                                "video_path": "/data/uploads/video.mp4",
                                "video_filename": "video.mp4",
                                "options": {},
                            }
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                )
            payload = json.loads(raised.exception.read().decode("utf-8"))
        finally:
            server.server_close()
            thread.join(timeout=1)

        self.assertFalse(payload["ok"])
        self.assertIn("Remote GPU runtime is unavailable", payload["error"])

    @patch("v2a_inspect_server.runtime.get_server_runtime_settings")
    def test_upload_endpoint_writes_raw_bytes(self, mock_server_settings) -> None:
        mock_server_settings.return_value = SimpleNamespace(shared_video_dir=Path("/tmp"))
        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            response = request.urlopen(
                request.Request(
                    f"http://127.0.0.1:{server.server_port}/upload",
                    data=b"video",
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": "5",
                        "X-Filename": "clip.mp4",
                    },
                    method="POST",
                )
            ).read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertTrue(str(payload["video_path"]).endswith(".mp4"))


if __name__ == "__main__":
    unittest.main()
