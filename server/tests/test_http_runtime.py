from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib import request
from urllib.error import HTTPError
from unittest.mock import patch

from v2a_inspect.clients.inference import (
    RemoteEmbeddingClient,
    RemoteLabelClient,
    RemoteSam3Client,
)
from v2a_inspect.tools.types import (
    EntityEmbedding,
    FrameBatch,
    LabelScore,
    Sam3EntityTrack,
    Sam3TrackPoint,
    Sam3TrackSet,
    SampledFrame,
)
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
        mock_check.return_value = SimpleNamespace(available=True, devices=[], minimum_vram_gb=16, message="ok")

        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            response = request.urlopen(f"http://127.0.0.1:{server.server_port}/healthz").read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "healthy")
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
        mock_check.return_value = SimpleNamespace(available=True, devices=[], minimum_vram_gb=10, message="ok")

        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            response = request.urlopen(f"http://127.0.0.1:{server.server_port}/readyz").read()
        finally:
            server.server_close()
            thread.join(timeout=1)

        payload = json.loads(response.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["bootstrap_ready"])
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
        mock_check.return_value = SimpleNamespace(available=True, devices=[], minimum_vram_gb=10, message="ok")

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

    def test_removed_full_pipeline_endpoints_return_gone(self) -> None:
        for path in ("/upload", "/analyze"):
            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                with self.assertRaises(HTTPError) as raised:
                    request.urlopen(
                        request.Request(
                            f"http://127.0.0.1:{server.server_port}{path}",
                            data=b"{}",
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                    )
                payload = json.loads(raised.exception.read().decode("utf-8"))
            finally:
                server.server_close()
                thread.join(timeout=1)
            self.assertIn("inference-only", payload["error"])

    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_sam3_inference_endpoint_returns_track_set(self, mock_build_tooling_runtime) -> None:
        fake_runtime = SimpleNamespace(
            sam3_client=SimpleNamespace(
                extract_entities=lambda frame_batches, **kwargs: Sam3TrackSet(
                    provider="sam3",
                    strategy="scene_prompt_seeded",
                    tracks=[
                        Sam3EntityTrack(
                            track_id="track-1",
                            scene_index=0,
                            start_seconds=0.0,
                            end_seconds=0.5,
                            confidence=0.9,
                            label_hint="paddle",
                            points=[Sam3TrackPoint(timestamp_seconds=0.0, frame_path=frame_batches[0].frames[0].image_path, confidence=0.9, bbox_xyxy=[0.0, 0.0, 10.0, 10.0])],
                        )
                    ],
                )
            )
        )
        mock_build_tooling_runtime.return_value = fake_runtime
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"frame-bytes")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                client = RemoteSam3Client(server_base_url=f"http://127.0.0.1:{server.server_port}")
                result = client.extract_entities(
                    [FrameBatch(scene_index=0, frames=[SampledFrame(scene_index=0, timestamp_seconds=0.0, image_path=str(image_path))])],
                    prompts_by_scene={0: ["paddle"]},
                )
            finally:
                server.server_close()
                thread.join(timeout=1)
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual(result.tracks[0].label_hint, "paddle")

    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_embed_crops_endpoint_returns_embeddings(self, mock_build_tooling_runtime) -> None:
        fake_runtime = SimpleNamespace(
            embedding_client=SimpleNamespace(
                embed_images=lambda image_paths_by_track: [
                    EntityEmbedding(
                        track_id=track_id,
                        model_name="fake",
                        vector=[float(index)],
                    )
                    for index, track_id in enumerate(sorted(image_paths_by_track))
                ]
            )
        )
        mock_build_tooling_runtime.return_value = fake_runtime
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "crop.jpg"
            image_path.write_bytes(b"crop-bytes")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                client = RemoteEmbeddingClient(server_base_url=f"http://127.0.0.1:{server.server_port}")
                result = client.embed_images({"track-1": [str(image_path)]})
            finally:
                server.server_close()
                thread.join(timeout=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].track_id, "track-1")

    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_score_labels_endpoint_returns_label_scores(self, mock_build_tooling_runtime) -> None:
        fake_runtime = SimpleNamespace(
            label_client=SimpleNamespace(
                score_image_labels=lambda *, image_paths, labels: [LabelScore(label=labels[0], score=0.9)]
            )
        )
        mock_build_tooling_runtime.return_value = fake_runtime
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "crop.jpg"
            image_path.write_bytes(b"crop-bytes")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                client = RemoteLabelClient(server_base_url=f"http://127.0.0.1:{server.server_port}")
                result = client.score_image_labels(image_paths=[str(image_path)], labels=["paddle"])
            finally:
                server.server_close()
                thread.join(timeout=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "paddle")

    @patch("v2a_inspect_server.runtime.build_tooling_runtime")
    def test_inference_endpoint_rejects_path_traversal_field_name(self, mock_build_tooling_runtime) -> None:
        mock_build_tooling_runtime.return_value = SimpleNamespace(
            embedding_client=SimpleNamespace(embed_images=lambda image_paths_by_track: [])
        )
        boundary = "----v2atestboundary"
        manifest = {
            "tracks": [
                {
                    "track_id": "track-1",
                    "images": [
                        {"upload_key": "../../evil", "filename": "crop.jpg"},
                    ],
                }
            ]
        }
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="manifest"\r\n'
            "Content-Type: application/json\r\n\r\n"
            f"{json.dumps(manifest)}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="../../evil"; filename="crop.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
            "crop-bytes\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler())
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            with self.assertRaises(HTTPError) as raised:
                request.urlopen(
                    request.Request(
                        f"http://127.0.0.1:{server.server_port}/infer/embed-crops",
                        data=body,
                        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                        method="POST",
                    )
                )
            payload = json.loads(raised.exception.read().decode("utf-8"))
        finally:
            server.server_close()
            thread.join(timeout=1)
        self.assertIn("Invalid multipart field name", payload["error"])


if __name__ == "__main__":
    unittest.main()
