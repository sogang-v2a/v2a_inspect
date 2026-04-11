from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from v2a_inspect.benchmark_samples import (
    collect_remote_artifact_paths,
    ensure_video_samples_extracted,
    load_benchmark_video_samples_manifest,
)


class BenchmarkSamplesTests(unittest.TestCase):
    def test_manifest_loads_all_tracked_video_samples(self) -> None:
        manifest = load_benchmark_video_samples_manifest(
            Path("docs/benchmark_video_samples_manifest.json")
        )
        self.assertEqual(manifest.version, "video-samples-v1")
        self.assertEqual(len(manifest.samples), 14)
        self.assertEqual(len({sample.filename for sample in manifest.samples}), 14)

    def test_ensure_video_samples_extracted_filters_mac_junk(self) -> None:
        manifest_payload = {
            "version": "test-v1",
            "samples": [
                {
                    "clip_id": "sample",
                    "filename": "sample.mp4",
                    "category": "test",
                    "expected_sources": [],
                    "event_notes": [],
                    "routing_notes": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            archive_path = root / "samples.zip"
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            with ZipFile(archive_path, "w") as zip_file:
                zip_file.writestr("video_samples/sample.mp4", b"video")
                zip_file.writestr("__MACOSX/video_samples/._sample.mp4", b"junk")
                zip_file.writestr("video_samples/.DS_Store", b"junk")
            manifest = load_benchmark_video_samples_manifest(manifest_path)
            extracted = ensure_video_samples_extracted(
                archive_path=archive_path,
                output_dir=root / "out",
                manifest=manifest,
            )
            self.assertEqual([path.name for path in extracted], ["sample.mp4"])
            self.assertEqual((root / "out" / "sample.mp4").read_bytes(), b"video")
            self.assertFalse((root / "out" / "._sample.mp4").exists())

    def test_collect_remote_artifact_paths_includes_runtime_trace(self) -> None:
        payload = {
            "multitrack_bundle": {
                "artifacts": {
                    "bundle_path": "/remote/run/bundle.json",
                    "storyboard_path": "/remote/run/storyboard.jpg",
                    "trace_path": "/remote/run/agent-trace.jsonl",
                },
                "pipeline_metadata": {
                    "runtime_trace_path": "/remote/run/runtime-trace.jsonl",
                },
            }
        }
        artifact_paths = collect_remote_artifact_paths(payload)
        self.assertEqual(
            artifact_paths["remote_runtime_trace_path"],
            "/remote/run/runtime-trace.jsonl",
        )
        self.assertEqual(
            artifact_paths["remote_storyboard_path"],
            "/remote/run/storyboard.jpg",
        )


if __name__ == "__main__":
    unittest.main()
