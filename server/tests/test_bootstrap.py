from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2a_inspect_server.bootstrap import (
    WeightsArtifact,
    WeightsBootstrapper,
    WeightsManifest,
)


class _FakeBinaryResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "_FakeBinaryResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class BootstrapTests(unittest.TestCase):
    def test_load_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "name": "sam3",
                                "repository": "facebook/sam3",
                                "filename": "sam3.safetensors",
                                "relative_path": "sam3/sam3.safetensors",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bootstrapper = WeightsBootstrapper(cache_dir=Path(tmp_dir))
            manifest = bootstrapper.load_manifest(manifest_path)
            self.assertEqual(len(manifest.artifacts), 1)

    @patch("v2a_inspect_server.bootstrap.download_file")
    def test_ensure_artifact_downloads_only_once(self, mock_download_file) -> None:
        def _fake_download(
            _url: str,
            destination: Path,
            *,
            api_key: str | None = None,
            timeout_seconds: int = 120,
        ) -> None:
            destination.write_bytes(b"weights")

        mock_download_file.side_effect = _fake_download
        with tempfile.TemporaryDirectory() as tmp_dir:
            bootstrapper = WeightsBootstrapper(
                cache_dir=Path(tmp_dir), hf_token="token"
            )
            artifact = WeightsArtifact(
                name="sam3",
                repository="facebook/sam3",
                filename="sam3.safetensors",
                relative_path="sam3/sam3.safetensors",
            )
            first = bootstrapper.ensure_artifact(artifact)
            second = bootstrapper.ensure_artifact(artifact)
            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), b"weights")
            self.assertEqual(mock_download_file.call_count, 1)

    def test_ensure_manifest_returns_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            bootstrapper = WeightsBootstrapper(cache_dir=cache_dir)
            artifact = cache_dir / "sam3/sam3.safetensors"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"ok")
            manifest = WeightsManifest(
                artifacts=[
                    WeightsArtifact(
                        name="sam3",
                        repository="facebook/sam3",
                        filename="sam3.safetensors",
                        relative_path="sam3/sam3.safetensors",
                    )
                ]
            )
            resolved = bootstrapper.ensure_manifest(manifest)
            self.assertEqual(resolved["sam3"], artifact)


if __name__ == "__main__":
    unittest.main()
