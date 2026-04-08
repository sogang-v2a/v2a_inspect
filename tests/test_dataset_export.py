from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2a_inspect.contracts import ArtifactRefs, MultitrackDescriptionBundle, ValidationReport, VideoMeta
from v2a_inspect.dataset import build_dataset_record, export_dataset_batch, export_dataset_record


class DatasetExportTests(unittest.TestCase):
    def test_build_and_export_dataset_record(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="vid-001",
            video_meta=VideoMeta(duration_seconds=3.0, fps=2.0, width=320, height=240),
            validation=ValidationReport(status="pass_with_warnings"),
            artifacts=ArtifactRefs(storyboard_dir="/tmp/storyboard", crop_dir="/tmp/crops"),
            pipeline_metadata={"pipeline_version": "stage7-test", "tool_versions": {"sam3": "1.0"}},
        )
        record = build_dataset_record(video_ref="video.mp4", bundle=bundle)
        self.assertEqual(record.pipeline_version, "stage7-test")
        with tempfile.TemporaryDirectory() as tmp_dir:
            exported = export_dataset_record(record, Path(tmp_dir) / "record.json")
            self.assertTrue(exported.exists())

    def test_export_dataset_batch_smoke(self) -> None:
        bundle = MultitrackDescriptionBundle(
            video_id="vid-001",
            video_meta=VideoMeta(duration_seconds=1.0, fps=2.0, width=320, height=240),
            validation=ValidationReport(status="pass"),
        )
        records = [
            build_dataset_record(video_ref=f"video-{index}.mp4", bundle=bundle.model_copy(update={"video_id": f"vid-{index}"}))
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            exported = export_dataset_batch(records, output_dir=tmp_dir)
            self.assertEqual(len(exported), 2)
            self.assertTrue(all(path.exists() for path in exported))
