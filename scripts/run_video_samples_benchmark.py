#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2a_inspect.benchmark_samples import (
    build_benchmark_run_id,
    ensure_video_samples_extracted,
    load_benchmark_video_samples_manifest,
    manifest_to_json,
    run_video_sample_benchmark,
    server_runtime_info,
    warmup_server,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local video sample benchmark pack against the forwarded sogang_gpu server."
    )
    parser.add_argument(
        "--archive",
        default="data/video_samples.zip",
        help="Path to the local zip archive of benchmark samples.",
    )
    parser.add_argument(
        "--samples-dir",
        default="data/video_samples",
        help="Directory where filtered mp4 files should be extracted.",
    )
    parser.add_argument(
        "--manifest",
        default="docs/benchmark_video_samples_manifest.json",
        help="Tracked manifest describing the local benchmark pack.",
    )
    parser.add_argument(
        "--output-root",
        default="data/benchmarks",
        help="Directory where local benchmark outputs should be saved.",
    )
    parser.add_argument(
        "--server-base-url",
        default="http://127.0.0.1:18080",
        help="Forwarded server base URL to benchmark against.",
    )
    parser.add_argument(
        "--ssh-host",
        default="sogang_gpu",
        help="SSH host alias used for optional artifact copies.",
    )
    parser.add_argument(
        "--no-copy-remote-artifacts",
        action="store_true",
        help="Skip copying remote trace/storyboard artifacts back to this machine.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip POST /warmup before running the sample pack.",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        default=[],
        help="Optional clip_id filter. May be repeated.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Benchmark mode to run. Defaults to both foundation and agentic.",
    )
    args = parser.parse_args()

    manifest = load_benchmark_video_samples_manifest(args.manifest)
    samples = ensure_video_samples_extracted(
        archive_path=args.archive,
        output_dir=args.samples_dir,
        manifest=manifest,
    )
    by_name = {path.name: path for path in samples}
    selected = [
        sample
        for sample in manifest.samples
        if not args.clip_id or sample.clip_id in set(args.clip_id)
    ]
    if not selected:
        raise SystemExit("No samples matched the requested clip_id filter.")

    modes = args.mode or ["tool_first_foundation", "agentic_tool_first"]
    run_id = build_benchmark_run_id()
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    _write_json(run_root / "manifest.json", manifest_to_json(manifest))
    _write_json(run_root / "runtime_info.json", server_runtime_info(args.server_base_url))
    if args.skip_warmup:
        warmup_payload: dict[str, Any] = {"skipped": True}
    else:
        warmup_payload = warmup_server(args.server_base_url)
    _write_json(run_root / "warmup.json", warmup_payload)

    summaries: list[dict[str, Any]] = []
    for sample in selected:
        video_path = by_name.get(sample.filename)
        if video_path is None:
            raise FileNotFoundError(f"Missing extracted video for {sample.filename}")
        for mode in modes:
            output_dir = run_root / sample.clip_id / mode
            summary = run_video_sample_benchmark(
                server_base_url=args.server_base_url,
                video_path=str(video_path),
                mode=mode,
                clip_id=sample.clip_id,
                output_dir=output_dir,
                ssh_host=None if args.no_copy_remote_artifacts else args.ssh_host,
            )
            summary["filename"] = sample.filename
            summary["category"] = sample.category
            summary["expected_sources"] = sample.expected_sources
            summary["event_notes"] = sample.event_notes
            summary["routing_notes"] = sample.routing_notes
            summaries.append(summary)
            print(
                json.dumps(
                    {
                        "clip_id": sample.clip_id,
                        "mode": mode,
                        "elapsed_seconds": summary["elapsed_seconds"],
                        "physical_source_count": summary["physical_source_count"],
                        "sound_event_count": summary["sound_event_count"],
                        "generation_group_count": summary["generation_group_count"],
                        "validation_status": summary["validation_status"],
                    }
                )
            )

    _write_json(
        run_root / "index.json",
        {
            "run_id": run_id,
            "server_base_url": args.server_base_url,
            "modes": modes,
            "sample_count": len(selected),
            "summaries": summaries,
        },
    )
    print(json.dumps({"run_root": str(run_root), "summary_count": len(summaries)}))
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
