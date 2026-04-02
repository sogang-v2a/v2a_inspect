from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from urllib import request as urllib_request

from v2a_inspect.runner import get_grouped_analysis, run_inspect
from v2a_inspect.settings import settings
from v2a_inspect.workflows import InspectOptions

from .bootstrap import WeightsBootstrapper, WeightsManifest
from .embeddings import EmbeddingClient, LabelClient
from .gpu_runtime import inspect_nvidia_runtime, runtime_check_to_json
from .sam3 import Sam3Client
from .tool_context import build_tool_context


@dataclass(frozen=True)
class ToolingRuntime:
    sam3_client: Sam3Client
    embedding_client: EmbeddingClient
    label_client: LabelClient
    bootstrapper: WeightsBootstrapper
    weights_manifest: WeightsManifest
    resolved_artifacts: dict[str, Path]


@lru_cache(maxsize=1)
def build_tooling_runtime() -> ToolingRuntime:
    bootstrapper = WeightsBootstrapper(
        cache_dir=Path(settings.model_cache_dir),
        hf_token=(
            settings.hf_token.get_secret_value()
            if settings.hf_token is not None
            else None
        ),
    )
    weights_manifest = bootstrapper.load_manifest(Path(settings.weights_manifest_path))
    resolved_artifacts = bootstrapper.resolve_manifest(weights_manifest)
    missing = [
        name for name, path in resolved_artifacts.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing bootstrapped model artifacts: " + ", ".join(sorted(missing))
        )
    return ToolingRuntime(
        sam3_client=Sam3Client(model_dir=resolved_artifacts["sam3"]),
        embedding_client=EmbeddingClient(model_dir=resolved_artifacts["embedding"]),
        label_client=LabelClient(model_dir=resolved_artifacts["label"]),
        bootstrapper=bootstrapper,
        weights_manifest=weights_manifest,
        resolved_artifacts=resolved_artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v2a-inspect-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("runtime-info", help="Show server runtime configuration")
    subparsers.add_parser(
        "bootstrap", help="Bootstrap model weights into the configured cache"
    )
    subparsers.add_parser(
        "check", help="Validate NVIDIA GPU visibility and minimum VRAM"
    )
    subparsers.add_parser("serve", help="Run the server runtime HTTP API")

    args = parser.parse_args(argv)
    if args.command == "runtime-info":
        return _run_runtime_info()
    if args.command == "bootstrap":
        return _run_bootstrap()
    if args.command == "serve":
        return _run_serve()
    return _run_check()


def _run_runtime_info() -> int:
    payload = {
        "runtime_mode": settings.runtime_mode,
        "model_cache_dir": str(settings.model_cache_dir),
        "weights_manifest_path": str(settings.weights_manifest_path),
        "minimum_gpu_vram_gb": settings.minimum_gpu_vram_gb,
        "server_bind_host": settings.server_bind_host,
        "server_bind_port": settings.server_bind_port,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_bootstrap() -> int:
    bootstrapper = WeightsBootstrapper(
        cache_dir=Path(settings.model_cache_dir),
        hf_token=(
            settings.hf_token.get_secret_value()
            if settings.hf_token is not None
            else None
        ),
    )
    manifest = bootstrapper.load_manifest(Path(settings.weights_manifest_path))
    resolved = bootstrapper.ensure_manifest(manifest)
    print(json.dumps({name: str(path) for name, path in resolved.items()}, indent=2))
    return 0


def _run_check() -> int:
    result = inspect_nvidia_runtime(minimum_vram_gb=settings.minimum_gpu_vram_gb)
    print(runtime_check_to_json(result))
    return 0 if result.available else 1


def _run_serve() -> int:
    server = ThreadingHTTPServer(
        (settings.server_bind_host, settings.server_bind_port),
        _build_handler(),
    )
    print(
        json.dumps(
            {
                "message": "v2a-inspect-server listening",
                "host": settings.server_bind_host,
                "port": settings.server_bind_port,
            }
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _build_handler() -> type[BaseHTTPRequestHandler]:
    class RuntimeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                result = inspect_nvidia_runtime(
                    minimum_vram_gb=settings.minimum_gpu_vram_gb
                )
                tooling_ready = True
                tooling_error = None
                try:
                    build_tooling_runtime()
                except Exception as exc:  # noqa: BLE001
                    tooling_ready = False
                    tooling_error = str(exc)
                self._write_json(
                    {
                        "ok": result.available and tooling_ready,
                        "runtime_mode": settings.runtime_mode,
                        "gpu_check": json.loads(runtime_check_to_json(result)),
                        "tooling_runtime_ready": tooling_ready,
                        "tooling_error": tooling_error,
                    }
                )
                return
            if self.path == "/runtime-info":
                payload = {
                    "runtime_mode": settings.runtime_mode,
                    "model_cache_dir": str(settings.model_cache_dir),
                    "weights_manifest_path": str(settings.weights_manifest_path),
                    "minimum_gpu_vram_gb": settings.minimum_gpu_vram_gb,
                    "server_bind_host": settings.server_bind_host,
                    "server_bind_port": settings.server_bind_port,
                }
                self._write_json(payload)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.startswith("/upload"):
                filename = self.headers.get("X-Filename", "video.mp4")
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    self.send_error(
                        HTTPStatus.BAD_REQUEST, "Upload body must not be empty."
                    )
                    return
                upload_bytes = self.rfile.read(content_length)
                upload_path = _write_uploaded_video(
                    filename=filename,
                    raw_bytes=upload_bytes,
                )
                self._write_json({"ok": True, "video_path": upload_path})
                return
            if self.path == "/bootstrap":
                try:
                    bootstrapper = WeightsBootstrapper(
                        cache_dir=Path(settings.model_cache_dir),
                        hf_token=(
                            settings.hf_token.get_secret_value()
                            if settings.hf_token is not None
                            else None
                        ),
                    )
                    manifest = bootstrapper.load_manifest(
                        Path(settings.weights_manifest_path)
                    )
                    resolved = bootstrapper.ensure_manifest(manifest)
                    build_tooling_runtime.cache_clear()
                    self._write_json(
                        {
                            "ok": True,
                            "artifacts": {
                                name: str(path) for name, path in resolved.items()
                            },
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    self._write_json(
                        {
                            "ok": False,
                            "error": str(exc),
                        },
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return
            if self.path == "/analyze":
                try:
                    payload = self._read_json()
                    video_path = payload.get("video_path")
                    video_filename = payload.get("video_filename")
                    video_url = payload.get("video_url")
                    options_payload = payload.get("options", {})
                    if not isinstance(video_path, str) and not isinstance(
                        video_url, str
                    ):
                        self.send_error(
                            HTTPStatus.BAD_REQUEST,
                            "video_path or video_url is required",
                        )
                        return
                    if not isinstance(options_payload, dict):
                        self.send_error(
                            HTTPStatus.BAD_REQUEST, "options must be an object"
                        )
                        return

                    options = InspectOptions.model_validate(options_payload)
                    resolved_video_path = _resolve_request_video_path(
                        video_path=video_path if isinstance(video_path, str) else "",
                        video_filename=video_filename
                        if isinstance(video_filename, str)
                        else "video.mp4",
                        video_url=video_url if isinstance(video_url, str) else None,
                    )
                    server_options = options.model_copy(
                        update={"runtime_mode": "in_process"}
                    )
                    tooling_runtime = build_tooling_runtime()
                    tool_context = build_tool_context(
                        resolved_video_path,
                        options=server_options,
                        tooling_runtime=tooling_runtime,
                    )
                    state = run_inspect(
                        resolved_video_path,
                        options=server_options,
                        initial_state_overrides=tool_context,
                    )
                    grouped = get_grouped_analysis(state)
                    self._write_json(
                        {
                            "scene_analysis": state["scene_analysis"].model_dump(
                                mode="json"
                            ),
                            "grouped_analysis": grouped.model_dump(mode="json"),
                            "warnings": state.get("warnings", []),
                            "progress_messages": state.get("progress_messages", []),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    self._write_json(
                        {
                            "ok": False,
                            "error": str(exc),
                        },
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return None

        def _read_json(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body) if body else {}
            if not isinstance(payload, dict):
                raise TypeError("Request payload must be a JSON object.")
            return payload

        def _write_json(
            self,
            payload: Mapping[str, object],
            *,
            status_code: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RuntimeHandler


def _resolve_request_video_path(
    *,
    video_path: str,
    video_filename: str,
    video_url: str | None,
) -> str:
    if video_path and Path(video_path).exists():
        return video_path
    if video_url:
        target_dir = _prepare_upload_dir()
        target_path = target_dir / _sanitize_filename(video_filename)
        urllib_request.urlretrieve(video_url, target_path)  # noqa: S310
        return str(target_path)
    raise ValueError("video_url is required when video_path is not accessible.")


def _write_uploaded_video(*, filename: str, raw_bytes: bytes) -> str:
    target_dir = _prepare_upload_dir()
    target_path = target_dir / _sanitize_filename(filename)
    target_path.write_bytes(raw_bytes)
    return str(target_path)


def _prepare_upload_dir() -> Path:
    target_root = settings.shared_video_dir or Path(tempfile.gettempdir())
    resolved_root = Path(target_root)
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        resolved_root = Path(tempfile.gettempdir())
        resolved_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix="v2a_inspect_server_upload_",
            dir=str(resolved_root),
        )
    )


def _sanitize_filename(filename: str) -> str:
    return (
        "".join(
            char for char in Path(filename).name if char.isalnum() or char in "._-"
        )
        or "video.mp4"
    )
