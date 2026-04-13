from __future__ import annotations

import argparse
import getpass
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.observability import build_cli_trace_context, flush_langfuse
from v2a_inspect.pipeline.prompt_templates import sync_prompts
from v2a_inspect.runner import get_multitrack_bundle, run_inspect
from v2a_inspect.settings import settings
from v2a_inspect.tools import detect_scenes, probe_video
from v2a_inspect.workflows import InspectOptions


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    finally:
        flush_langfuse()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v2a-inspect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_defaults = InspectOptions()
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run the supported silent-video inspect workflow for a video",
    )
    analyze_parser.add_argument("video_path", help="Path to the input video")
    _add_analyze_option_arguments(analyze_parser, defaults=analyze_defaults)
    analyze_parser.add_argument(
        "-o",
        "--output",
        help="Write MultitrackDescriptionBundle JSON to this path instead of stdout",
    )
    analyze_parser.set_defaults(func=_run_analyze_command)

    probe_parser = subparsers.add_parser(
        "probe",
        help="Probe video metadata for the tool-first visual pipeline",
    )
    probe_parser.add_argument("video_path", help="Path to the input video")
    probe_parser.add_argument(
        "-o",
        "--output",
        help="Write probe JSON to this path instead of stdout",
    )
    probe_parser.set_defaults(func=_run_probe_command)

    plan_scenes_parser = subparsers.add_parser(
        "plan-scenes",
        help="Build scene chunks for the tool-first visual pipeline",
    )
    plan_scenes_parser.add_argument("video_path", help="Path to the input video")
    plan_scenes_parser.add_argument(
        "--scene-seconds",
        type=float,
        default=5.0,
        help="Target chunk length used by the fixed-window scene planner",
    )
    plan_scenes_parser.add_argument(
        "-o",
        "--output",
        help="Write scene plan JSON to this path instead of stdout",
    )
    plan_scenes_parser.set_defaults(func=_run_plan_scenes_command)

    prompts_parser = subparsers.add_parser(
        "prompts",
        help="Manage Langfuse prompt definitions",
    )
    prompts_subparsers = prompts_parser.add_subparsers(
        dest="prompts_command",
        required=True,
    )
    sync_parser = prompts_subparsers.add_parser(
        "sync",
        help="Sync local prompts to Langfuse prompt management",
    )
    sync_parser.add_argument(
        "--label",
        default=settings.langfuse_prompt_label,
        help="Langfuse label to assign to synced prompts",
    )
    sync_parser.set_defaults(func=_run_prompt_sync_command)

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
    _add_common_runtime_arguments(parser, defaults=defaults)


def _add_common_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    defaults: InspectOptions,
) -> None:
    parser.add_argument(
        "--pipeline-mode",
        choices=("tool_first_foundation", "agentic_tool_first"),
        default=defaults.pipeline_mode,
    )
    parser.add_argument("--gemini-model", default=defaults.gemini_model)
    parser.add_argument(
        "--text-timeout-ms",
        type=int,
        default=defaults.text_timeout_ms,
    )
    parser.add_argument("--max-retries", type=int, default=defaults.max_retries)


def _run_analyze_command(args: argparse.Namespace) -> int:
    options = _build_analyze_options(args)
    trace_context = build_cli_trace_context(
        "analyze",
        user_id=_resolve_cli_user(),
        metadata={
            "video_path": args.video_path,
            "output_path": args.output,
            "scene_analysis_mode": options.scene_analysis_mode,
            "fps": options.fps,
        },
        tags=_build_runtime_tags(options),
    )
    state = run_inspect(
        args.video_path,
        options=options,
        progress_callback=_print_progress,
        warning_callback=_print_warning,
        trace_context=trace_context,
    )
    _print_trace_id(state)
    _write_bundle_json(get_multitrack_bundle(state), output_path=args.output)
    return 0


def _run_prompt_sync_command(args: argparse.Namespace) -> int:
    synced_prompts = sync_prompts(label=args.label)
    for prompt in synced_prompts:
        print(
            f"Synced prompt '{prompt.name}' to Langfuse with label '{args.label}'.",
            file=sys.stderr,
        )
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


def _run_probe_command(args: argparse.Namespace) -> int:
    video_probe = probe_video(args.video_path)
    _write_json_payload(
        video_probe.model_dump_json(indent=2) + "\n", output_path=args.output
    )
    return 0


def _run_plan_scenes_command(args: argparse.Namespace) -> int:
    video_probe = probe_video(args.video_path)
    scene_plan = detect_scenes(
        args.video_path,
        probe=video_probe,
        target_scene_seconds=args.scene_seconds,
    )
    payload = {
        "probe": video_probe.model_dump(mode="json"),
        "scenes": [scene.model_dump(mode="json") for scene in scene_plan],
    }
    _write_json_payload(
        json.dumps(payload, indent=2) + "\n",
        output_path=args.output,
    )
    return 0


def _build_analyze_options(args: argparse.Namespace) -> InspectOptions:
    return InspectOptions(
        fps=args.fps,
        pipeline_mode=args.pipeline_mode,
        scene_analysis_mode=args.scene_analysis_mode,
        gemini_model=args.gemini_model,
        text_timeout_ms=args.text_timeout_ms,
        max_retries=args.max_retries,
    )


def _write_bundle_json(
    bundle: MultitrackDescriptionBundle, *, output_path: str | None
) -> None:
    payload = bundle.model_dump_json(indent=2) + "\n"
    _write_json_payload(payload, output_path=output_path)


def _write_json_payload(payload: str, *, output_path: str | None) -> None:
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(payload, encoding="utf-8")
        print(f"Wrote JSON output to {output_file}", file=sys.stderr)
        return

    sys.stdout.write(payload)


def _print_progress(message: str) -> None:
    print(f"[progress] {message}", file=sys.stderr)


def _print_warning(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr)


def _print_trace_id(state: Mapping[str, object]) -> None:
    trace_id = state.get("trace_id")
    if trace_id:
        print(f"[trace] {trace_id}", file=sys.stderr)


def _build_runtime_tags(options: InspectOptions) -> list[str]:
    return [f"pipeline:{options.pipeline_mode}"]


def _resolve_cli_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown-user"
