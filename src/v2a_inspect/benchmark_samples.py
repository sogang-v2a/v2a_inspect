from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import request
from zipfile import ZipFile

from v2a_inspect.clients.server import run_server_inspect_raw
from v2a_inspect.workflows import InspectOptions


@dataclass(frozen=True)
class BenchmarkVideoSample:
    clip_id: str
    filename: str
    category: str
    expected_sources: list[str]
    event_notes: list[str]
    routing_notes: list[str]


@dataclass(frozen=True)
class BenchmarkVideoSamplesManifest:
    version: str
    samples: list[BenchmarkVideoSample]


def load_benchmark_video_samples_manifest(
    path: str | Path,
) -> BenchmarkVideoSamplesManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [
        BenchmarkVideoSample(
            clip_id=str(item["clip_id"]),
            filename=str(item["filename"]),
            category=str(item["category"]),
            expected_sources=[str(entry) for entry in item.get("expected_sources", [])],
            event_notes=[str(entry) for entry in item.get("event_notes", [])],
            routing_notes=[str(entry) for entry in item.get("routing_notes", [])],
        )
        for item in payload.get("samples", [])
    ]
    return BenchmarkVideoSamplesManifest(
        version=str(payload["version"]),
        samples=samples,
    )


def ensure_video_samples_extracted(
    *,
    archive_path: str | Path,
    output_dir: str | Path,
    manifest: BenchmarkVideoSamplesManifest,
) -> list[Path]:
    archive = Path(archive_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    filenames = {sample.filename for sample in manifest.samples}
    extracted: list[Path] = []
    with ZipFile(archive) as zip_file:
        members = {
            Path(member).name: member
            for member in zip_file.namelist()
            if _is_valid_sample_member(member)
        }
        for filename in sorted(filenames):
            member = members.get(filename)
            if member is None:
                raise FileNotFoundError(f"{filename} is missing from {archive}")
            target = output_root / filename
            if not target.exists():
                with zip_file.open(member, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def collect_remote_artifact_paths(response_payload: dict[str, Any]) -> dict[str, str]:
    bundle = response_payload.get("multitrack_bundle", {})
    artifacts = bundle.get("artifacts", {}) if isinstance(bundle, dict) else {}
    metadata = bundle.get("pipeline_metadata", {}) if isinstance(bundle, dict) else {}
    artifact_paths: dict[str, str] = {}
    _add_remote_path(artifact_paths, "remote_bundle_path", artifacts.get("bundle_path"))
    _add_remote_path(
        artifact_paths, "remote_storyboard_path", artifacts.get("storyboard_path")
    )
    _add_remote_path(artifact_paths, "remote_trace_path", artifacts.get("trace_path"))
    _add_remote_path(
        artifact_paths,
        "remote_runtime_trace_path",
        metadata.get("runtime_trace_path"),
    )
    _add_remote_path(
        artifact_paths,
        "remote_agent_review_trace_path",
        metadata.get("agent_review_trace_path"),
    )
    return artifact_paths


def copy_remote_artifacts(
    *,
    ssh_host: str,
    remote_artifact_paths: dict[str, str],
    output_dir: str | Path,
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, remote_path in sorted(remote_artifact_paths.items()):
        suffix = Path(remote_path).suffix or ".bin"
        target = output_root / f"{key}{suffix}"
        subprocess.run(
            ["scp", f"{ssh_host}:{remote_path}", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        copied[key] = str(target)
    return copied


def server_runtime_info(server_base_url: str) -> dict[str, Any]:
    with request.urlopen(f"{server_base_url.rstrip('/')}/runtime-info", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("runtime-info response must be a JSON object")
    return payload


def warmup_server(server_base_url: str) -> dict[str, Any]:
    request_obj = request.Request(
        url=f"{server_base_url.rstrip('/')}/warmup",
        headers={"Content-Type": "application/json"},
        data=b"{}",
        method="POST",
    )
    with request.urlopen(request_obj, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("warmup response must be a JSON object")
    return payload


def run_video_sample_benchmark(
    *,
    server_base_url: str,
    video_path: str,
    mode: str,
    clip_id: str,
    output_dir: str | Path,
    ssh_host: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    request_payload = {
        "clip_id": clip_id,
        "mode": mode,
        "video_path": video_path,
        "server_base_url": server_base_url,
    }
    _write_json(output_root / "request.json", request_payload)

    options = InspectOptions(
        runtime_mode="nvidia_docker",
        server_base_url=server_base_url,
        pipeline_mode=mode,  # type: ignore[arg-type]
    )
    started = perf_counter()
    response_payload = run_server_inspect_raw(
        server_base_url=server_base_url,
        video_path=video_path,
        options=options,
    )
    elapsed = round(perf_counter() - started, 4)
    _write_json(output_root / "response.json", response_payload)

    bundle = response_payload["multitrack_bundle"]
    _write_json(output_root / "bundle.json", bundle)
    remote_artifact_paths = collect_remote_artifact_paths(response_payload)
    copied_artifacts: dict[str, str] = {}
    if ssh_host:
        copied_artifacts = copy_remote_artifacts(
            ssh_host=ssh_host,
            remote_artifact_paths=remote_artifact_paths,
            output_dir=output_root / "artifacts",
        )

    summary = {
        "clip_id": clip_id,
        "mode": mode,
        "elapsed_seconds": elapsed,
        "physical_source_count": len(bundle.get("physical_sources", [])),
        "sound_event_count": len(bundle.get("sound_events", [])),
        "generation_group_count": len(bundle.get("generation_groups", [])),
        "validation_status": bundle.get("validation", {}).get("status"),
        "warnings": list(response_payload.get("warnings", [])),
        "progress_messages": list(response_payload.get("progress_messages", [])),
        "effective_runtime_profile": response_payload.get("effective_runtime_profile"),
        "runtime_profile_source": response_payload.get("runtime_profile_source"),
        "residency_mode": response_payload.get("residency_mode"),
        "warm_start": response_payload.get("warm_start"),
        "resident_models_before_run": list(
            response_payload.get("resident_models_before_run", [])
        ),
        "resident_models_after_run": list(
            response_payload.get("resident_models_after_run", [])
        ),
        "remote_artifact_paths": remote_artifact_paths,
        "copied_artifacts": copied_artifacts,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def build_benchmark_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def manifest_to_json(manifest: BenchmarkVideoSamplesManifest) -> dict[str, Any]:
    return {
        "version": manifest.version,
        "samples": [asdict(sample) for sample in manifest.samples],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _add_remote_path(target: dict[str, str], key: str, value: Any) -> None:
    if isinstance(value, str) and value:
        target[key] = value


def _is_valid_sample_member(member: str) -> bool:
    name = Path(member).name
    if not name:
        return False
    if member.startswith("__MACOSX/"):
        return False
    if name == ".DS_Store":
        return False
    if name.startswith("._"):
        return False
    return name.lower().endswith(".mp4")
