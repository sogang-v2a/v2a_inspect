from __future__ import annotations

from pathlib import Path

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.dataset.records import DatasetRecord


def build_dataset_record(
    *,
    video_ref: str,
    bundle: MultitrackDescriptionBundle,
    pipeline_version: str | None = None,
    model_versions: dict[str, str] | None = None,
    crop_evidence_enabled: bool = True,
) -> DatasetRecord:
    resolved_pipeline_version = pipeline_version or str(
        bundle.pipeline_metadata.get("pipeline_version", "unknown")
    )
    return DatasetRecord(
        record_id=f"{bundle.video_id}:{resolved_pipeline_version}",
        video_ref=video_ref,
        bundle=bundle,
        evidence_refs={
            "storyboard_dir": bundle.artifacts.storyboard_dir,
            "crop_dir": bundle.artifacts.crop_dir,
            "clip_dir": bundle.artifacts.clip_dir,
        },
        review_metadata=bundle.review_metadata.model_dump(mode="json"),
        pipeline_version=resolved_pipeline_version,
        model_versions=model_versions
        or {
            key: str(value)
            for key, value in bundle.pipeline_metadata.get("tool_versions", {}).items()
        },
        validation_status=bundle.validation.status,
        crop_evidence_enabled=crop_evidence_enabled,
    )


def export_dataset_record(record: DatasetRecord, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def export_dataset_batch(
    records: list[DatasetRecord],
    *,
    output_dir: str | Path,
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for record in records:
        exported.append(
            export_dataset_record(record, output_root / f"{record.record_id}.json")
        )
    return exported
