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
                                "relative_path": "sam3",
                                "allow_patterns": ["*.json"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bootstrapper = WeightsBootstrapper(cache_dir=Path(tmp_dir))
            manifest = bootstrapper.load_manifest(manifest_path)
            self.assertEqual(len(manifest.artifacts), 1)

    @patch("v2a_inspect_server.bootstrap.snapshot_download")
    def test_ensure_artifact_downloads_only_once(self, mock_snapshot_download) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bootstrapper = WeightsBootstrapper(
                cache_dir=Path(tmp_dir), hf_token="token"
            )
            artifact = WeightsArtifact(
                name="sam3",
                repository="facebook/sam3",
                relative_path="sam3",
                allow_patterns=["*.json"],
            )

            def _fake_snapshot_download(**kwargs):
                Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
                (Path(kwargs["local_dir"]) / "config.json").write_text("{}", encoding="utf-8")
                return str(kwargs["local_dir"])

            mock_snapshot_download.side_effect = _fake_snapshot_download
            first = bootstrapper.ensure_artifact(artifact)
            second = bootstrapper.ensure_artifact(artifact)
            self.assertEqual(first, second)
            self.assertTrue((first / "config.json").exists())
            self.assertEqual(mock_snapshot_download.call_count, 1)

    def test_ensure_manifest_returns_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            bootstrapper = WeightsBootstrapper(cache_dir=cache_dir)
            artifact_dir = cache_dir / "sam3"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "config.json").write_text("{}", encoding="utf-8")
            manifest = WeightsManifest(
                artifacts=[
                    WeightsArtifact(
                        name="sam3",
                        repository="facebook/sam3",
                        relative_path="sam3",
                    )
                ]
            )
            resolved = bootstrapper.ensure_manifest(manifest)
            self.assertEqual(resolved["sam3"], artifact_dir)


if __name__ == "__main__":
    unittest.main()
