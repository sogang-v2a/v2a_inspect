from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from v2a_inspect.pipeline.response_models import GroupedAnalysis, VideoSceneAnalysis
from v2a_inspect.runner import (
    get_grouped_analysis,
    run_group_from_scene_analysis,
    run_inspect,
)
from v2a_inspect.workflows import InspectOptions


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v2a-inspect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_defaults = InspectOptions()
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run the full inspect workflow for a video",
    )
    analyze_parser.add_argument("video_path", help="Path to the input video")
    _add_analyze_option_arguments(analyze_parser, defaults=analyze_defaults)
    analyze_parser.add_argument(
        "-o",
        "--output",
        help="Write GroupedAnalysis JSON to this path instead of stdout",
    )
    analyze_parser.set_defaults(func=_run_analyze_command)

    group_defaults = InspectOptions(enable_vlm_verify=False, enable_model_select=False)
    group_parser = subparsers.add_parser(
        "group",
        help="Run grouping from a precomputed VideoSceneAnalysis JSON file",
    )
    group_parser.add_argument(
        "scene_analysis_path",
        help="Path to a VideoSceneAnalysis JSON file",
    )
    group_parser.add_argument(
        "--video-path",
        help="Original video path, required for VLM verify or model selection",
    )
    _add_group_option_arguments(group_parser, defaults=group_defaults)
    group_parser.add_argument(
        "-o",
        "--output",
        help="Write GroupedAnalysis JSON to this path instead of stdout",
    )
    group_parser.set_defaults(func=_run_group_command)

    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Streamlit UI",
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8501)
    ui_parser.set_defaults(func=_run_ui_command)

    return parser


def _add_analyze_option_arguments(
    parser: argparse.ArgumentParser,
    *,
    defaults: InspectOptions,
) -> None:
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument(
        "--scene-analysis-mode",
        choices=("default", "extended"),
        default=defaults.scene_analysis_mode,
    )
    parser.add_argument(
        "--vlm-verify",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_vlm_verify,
        dest="enable_vlm_verify",
    )
    parser.add_argument(
        "--model-select",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_model_select,
        dest="enable_model_select",
    )
    _add_common_runtime_arguments(parser, defaults=defaults)


def _add_group_option_arguments(
    parser: argparse.ArgumentParser,
    *,
    defaults: InspectOptions,
) -> None:
    parser.add_argument("--fps", type=float, default=defaults.fps)
    parser.add_argument(
        "--vlm-verify",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_vlm_verify,
        dest="enable_vlm_verify",
    )
    parser.add_argument(
        "--model-select",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_model_select,
        dest="enable_model_select",
    )
    _add_common_runtime_arguments(parser, defaults=defaults)


def _add_common_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    defaults: InspectOptions,
) -> None:
    parser.add_argument("--gemini-model", default=defaults.gemini_model)
    parser.add_argument(
        "--upload-timeout-seconds",
        type=int,
        default=defaults.upload_timeout_seconds,
    )
    parser.add_argument(
        "--text-timeout-ms",
        type=int,
        default=defaults.text_timeout_ms,
    )
    parser.add_argument(
        "--video-timeout-ms",
        type=int,
        default=defaults.video_timeout_ms,
    )
    parser.add_argument("--max-retries", type=int, default=defaults.max_retries)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=defaults.poll_interval_seconds,
    )


def _run_analyze_command(args: argparse.Namespace) -> int:
    options = _build_analyze_options(args)
    state = run_inspect(
        args.video_path,
        options=options,
        progress_callback=_print_progress,
        warning_callback=_print_warning,
    )
    _write_grouped_analysis_json(get_grouped_analysis(state), output_path=args.output)
    return 0


def _run_group_command(args: argparse.Namespace) -> int:
    options = _build_group_options(args)
    if (
        options.enable_vlm_verify or options.enable_model_select
    ) and not args.video_path:
        raise SystemExit(
            "--video-path is required when --vlm-verify or --model-select is enabled."
        )

    scene_analysis = _load_scene_analysis(args.scene_analysis_path)
    state = run_group_from_scene_analysis(
        scene_analysis,
        options=options,
        video_path=args.video_path or "",
        progress_callback=_print_progress,
        warning_callback=_print_warning,
    )
    _write_grouped_analysis_json(get_grouped_analysis(state), output_path=args.output)
    return 0


def _run_ui_command(args: argparse.Namespace) -> int:
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    if not app_path.exists():
        raise SystemExit(f"The package Streamlit app was not found at {app_path}.")

    streamlit_executable = shutil.which("streamlit")
    if streamlit_executable is None:
        raise SystemExit(
            "The 'streamlit' executable is not available in this environment."
        )

    completed = subprocess.run(
        [
            streamlit_executable,
            "run",
            str(app_path),
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ],
        check=False,
    )
    return completed.returncode


def _build_analyze_options(args: argparse.Namespace) -> InspectOptions:
    return InspectOptions(
        fps=args.fps,
        scene_analysis_mode=args.scene_analysis_mode,
        enable_vlm_verify=args.enable_vlm_verify,
        enable_model_select=args.enable_model_select,
        gemini_model=args.gemini_model,
        upload_timeout_seconds=args.upload_timeout_seconds,
        text_timeout_ms=args.text_timeout_ms,
        video_timeout_ms=args.video_timeout_ms,
        max_retries=args.max_retries,
        poll_interval_seconds=args.poll_interval_seconds,
    )


def _build_group_options(args: argparse.Namespace) -> InspectOptions:
    return InspectOptions(
        fps=args.fps,
        scene_analysis_mode="default",
        enable_vlm_verify=args.enable_vlm_verify,
        enable_model_select=args.enable_model_select,
        gemini_model=args.gemini_model,
        upload_timeout_seconds=args.upload_timeout_seconds,
        text_timeout_ms=args.text_timeout_ms,
        video_timeout_ms=args.video_timeout_ms,
        max_retries=args.max_retries,
        poll_interval_seconds=args.poll_interval_seconds,
    )


def _load_scene_analysis(scene_analysis_path: str) -> VideoSceneAnalysis:
    payload = Path(scene_analysis_path).read_text(encoding="utf-8")
    return VideoSceneAnalysis.model_validate_json(payload)


def _write_grouped_analysis_json(
    grouped_analysis: GroupedAnalysis, *, output_path: str | None
) -> None:
    payload = grouped_analysis.model_dump_json(indent=2) + "\n"
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(payload, encoding="utf-8")
        print(f"Wrote grouped analysis to {output_file}", file=sys.stderr)
        return

    sys.stdout.write(payload)


def _print_progress(message: str) -> None:
    print(f"[progress] {message}", file=sys.stderr)


def _print_warning(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr)
