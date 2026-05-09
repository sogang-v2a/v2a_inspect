from __future__ import annotations

import argparse
import cgi
import json
import re
import tempfile
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qs, urlparse

from v2a_inspect.remote_inference_payloads import (
    EmbedImagesManifest,
    LabelScoreManifest,
    Sam3ExtractManifest,
)
from v2a_inspect.tools.types import FrameBatch, SampledFrame

from .bootstrap import WeightsBootstrapper, WeightsManifest
from .gpu_runtime import inspect_nvidia_runtime, runtime_check_to_json
from .settings import get_server_runtime_settings

if TYPE_CHECKING:
    from .embeddings import EmbeddingClient, LabelClient
    from .sam3 import Sam3Client


_MAX_MULTIPART_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_MULTIPART_FILE_BYTES = 8 * 1024 * 1024
_MAX_MULTIPART_FILE_COUNT = 128
_SAFE_MULTIPART_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


class ToolingRuntime:
    def __init__(
        self,
        *,
        bootstrapper: WeightsBootstrapper,
        weights_manifest: WeightsManifest,
        resolved_artifacts: dict[str, Path],
        runtime_profile: Literal["mig10_safe", "full_gpu", "cpu_dev"],
    ) -> None:
        self.bootstrapper = bootstrapper
        self.weights_manifest = weights_manifest
        self.resolved_artifacts = resolved_artifacts
        self.runtime_profile = runtime_profile
        self._sam3_client: Sam3Client | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._label_client: LabelClient | None = None

    @property
    def should_release_clients(self) -> bool:
        return self.runtime_profile == "mig10_safe"

    @property
    def residency_mode(self) -> Literal["resident", "release_after_stage"]:
        return "release_after_stage" if self.should_release_clients else "resident"

    @property
    def sam3_client(self) -> Sam3Client:
        if self._sam3_client is None:
            from .sam3 import Sam3Client

            self._sam3_client = Sam3Client(model_dir=self.resolved_artifacts["sam3"])
        return self._sam3_client

    @property
    def embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            from .embeddings import EmbeddingClient

            self._embedding_client = EmbeddingClient(model_dir=self.resolved_artifacts["embedding"])
        return self._embedding_client

    @property
    def label_client(self) -> LabelClient:
        if self._label_client is None:
            from .embeddings import LabelClient

            self._label_client = LabelClient(model_dir=self.resolved_artifacts["label"])
        return self._label_client

    def artifacts_missing(self) -> list[str]:
        return [name for name, path in self.resolved_artifacts.items() if not path.exists()]

    def resident_client_names(self) -> list[str]:
        clients: list[str] = []
        if self._sam3_client is not None:
            clients.append("sam3")
        if self._embedding_client is not None:
            clients.append("embedding")
        if self._label_client is not None:
            clients.append("label")
        return clients

    def warmup_visual_clients(self) -> dict[str, object]:
        timings: dict[str, float] = {}
        status: dict[str, str] = {}
        for client_name in ("sam3", "embedding", "label"):
            started = perf_counter()
            try:
                getattr(self, f"{client_name}_client")
                timings[client_name] = round(perf_counter() - started, 4)
                status[client_name] = "ready"
            except Exception as exc:  # noqa: BLE001
                timings[client_name] = round(perf_counter() - started, 4)
                status[client_name] = f"error: {exc}"
                raise
        return {
            "model_load_status": status,
            "model_load_seconds": timings,
            "resident_models": self.resident_client_names(),
        }


@lru_cache(maxsize=1)
def build_tooling_runtime() -> ToolingRuntime:
    server_settings = get_server_runtime_settings()
    bootstrapper = WeightsBootstrapper(
        cache_dir=Path(server_settings.model_cache_dir),
        hf_token=server_settings.hf_token,
    )
    weights_manifest = bootstrapper.load_manifest(Path(server_settings.weights_manifest_path))
    if not weights_manifest.artifacts:
        raise FileNotFoundError(
            f"No model artifacts are defined in {server_settings.weights_manifest_path}."
        )
    resolved_artifacts = bootstrapper.resolve_manifest(weights_manifest)
    missing = [name for name, path in resolved_artifacts.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing bootstrapped model artifacts: " + ", ".join(sorted(missing)))
    return ToolingRuntime(
        bootstrapper=bootstrapper,
        weights_manifest=weights_manifest,
        resolved_artifacts=resolved_artifacts,
        runtime_profile=server_settings.runtime_profile,
    )


def _runtime_info_payload() -> tuple[dict[str, object], HTTPStatus]:
    server_settings = get_server_runtime_settings()
    payload: dict[str, object] = {
        "runtime_mode": server_settings.runtime_mode,
        "runtime_profile": server_settings.runtime_profile,
        "effective_runtime_profile": server_settings.runtime_profile,
        "runtime_profile_source": "server_settings",
        "remote_gpu_target": server_settings.remote_gpu_target,
        "model_cache_dir": str(server_settings.model_cache_dir),
        "weights_manifest_path": str(server_settings.weights_manifest_path),
        "minimum_gpu_vram_gb": server_settings.minimum_gpu_vram_gb,
        "server_bind_host": server_settings.server_bind_host,
        "server_bind_port": server_settings.server_bind_port,
    }
    try:
        tooling_runtime = build_tooling_runtime()
        payload.update(
            {
                "tooling_runtime_ready": not tooling_runtime.artifacts_missing(),
                "tooling_error": None,
                "residency_mode": tooling_runtime.residency_mode,
                "resident_models": tooling_runtime.resident_client_names(),
                "supported_inference_endpoints": [
                    "/infer/sam3-extract",
                    "/infer/embed-crops",
                    "/infer/score-labels",
                ],
                "removed_endpoints": ["/upload", "/analyze"],
            }
        )
        return payload, HTTPStatus.OK
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "tooling_runtime_ready": False,
                "tooling_error": str(exc),
                "residency_mode": "unknown",
                "resident_models": [],
                "supported_inference_endpoints": [
                    "/infer/sam3-extract",
                    "/infer/embed-crops",
                    "/infer/score-labels",
                ],
                "removed_endpoints": ["/upload", "/analyze"],
            }
        )
        return payload, HTTPStatus.OK


def _readyz_payload(
    *, include_model_load_check: bool = False
) -> tuple[dict[str, object], HTTPStatus]:
    server_settings = get_server_runtime_settings()
    gpu_check = inspect_nvidia_runtime(minimum_vram_gb=server_settings.minimum_gpu_vram_gb)
    payload: dict[str, object] = {
        "runtime_mode": server_settings.runtime_mode,
        "runtime_profile": server_settings.runtime_profile,
        "remote_gpu_target": server_settings.remote_gpu_target,
        "gpu_check": json.loads(runtime_check_to_json(gpu_check)),
    }
    if not gpu_check.available and server_settings.runtime_profile != "cpu_dev":
        payload.update(
            {
                "ok": False,
                "bootstrap_ready": False,
                "tooling_runtime_ready": False,
                "tooling_error": "Remote GPU runtime is unavailable.",
            }
        )
        return payload, HTTPStatus.SERVICE_UNAVAILABLE
    try:
        tooling_runtime = build_tooling_runtime()
        missing = tooling_runtime.artifacts_missing()
        payload.update(
            {
                "ok": not missing,
                "bootstrap_ready": not missing,
                "missing_artifacts": missing,
                "tooling_runtime_ready": not missing,
                "tooling_error": None,
                "resident_models": tooling_runtime.resident_client_names(),
                "residency_mode": tooling_runtime.residency_mode,
                "warnings": (["The load_models query flag is deprecated; use POST /warmup for persistent model warmup."] if include_model_load_check else []),
            }
        )
        return payload, HTTPStatus.OK if not missing else HTTPStatus.SERVICE_UNAVAILABLE
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "ok": False,
                "bootstrap_ready": False,
                "tooling_runtime_ready": False,
                "tooling_error": str(exc),
                "resident_models": [],
                "residency_mode": "unknown",
                "warnings": [],
            }
        )
        return payload, HTTPStatus.SERVICE_UNAVAILABLE


def _warmup_payload(
    *,
    source: Literal["warmup_endpoint", "readyz_alias"] = "warmup_endpoint",
) -> tuple[dict[str, object], HTTPStatus]:
    payload, status = _readyz_payload()
    if status != HTTPStatus.OK:
        payload["warmup_ok"] = False
        payload["warmup_source"] = source
        return payload, status
    tooling_runtime = build_tooling_runtime()
    try:
        warmup = tooling_runtime.warmup_visual_clients()
    except Exception as exc:  # noqa: BLE001
        payload.update({"warmup_ok": False, "warmup_source": source, "error": str(exc)})
        return payload, HTTPStatus.INTERNAL_SERVER_ERROR
    payload.update(
        {
            "warmup_ok": True,
            "warmup_source": source,
            "model_load_status": warmup["model_load_status"],
            "model_load_seconds": warmup["model_load_seconds"],
            "resident_models": warmup["resident_models"],
        }
    )
    return payload, HTTPStatus.OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v2a-inspect-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("runtime-info", help="Show server runtime configuration")
    subparsers.add_parser("bootstrap", help="Bootstrap model weights into the configured cache")
    subparsers.add_parser("check", help="Validate NVIDIA GPU visibility and minimum VRAM")
    subparsers.add_parser("warmup", help="Load visual models and keep them resident")
    subparsers.add_parser("serve", help="Run the server runtime HTTP API")

    args = parser.parse_args(argv)
    if args.command == "runtime-info":
        return _run_runtime_info()
    if args.command == "bootstrap":
        return _run_bootstrap()
    if args.command == "serve":
        return _run_serve()
    if args.command == "warmup":
        return _run_warmup()
    return _run_check()


def _run_runtime_info() -> int:
    payload, _ = _runtime_info_payload()
    print(json.dumps(payload, indent=2))
    return 0


def _run_warmup() -> int:
    payload, status = _warmup_payload()
    print(json.dumps(payload, indent=2))
    return 0 if status == HTTPStatus.OK else 1


def _run_bootstrap() -> int:
    server_settings = get_server_runtime_settings()
    bootstrapper = WeightsBootstrapper(
        cache_dir=Path(server_settings.model_cache_dir),
        hf_token=server_settings.hf_token,
    )
    manifest = bootstrapper.load_manifest(Path(server_settings.weights_manifest_path))
    resolved = bootstrapper.ensure_manifest(manifest)
    print(json.dumps({name: str(path) for name, path in resolved.items()}, indent=2))
    return 0


def _run_check() -> int:
    server_settings = get_server_runtime_settings()
    check = inspect_nvidia_runtime(minimum_vram_gb=server_settings.minimum_gpu_vram_gb)
    print(runtime_check_to_json(check))
    return 0 if (check.available or server_settings.runtime_profile == "cpu_dev") else 1


def _run_serve() -> int:
    server_settings = get_server_runtime_settings()
    httpd = ThreadingHTTPServer(
        (server_settings.server_bind_host, server_settings.server_bind_port),
        _build_handler(),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "host": server_settings.server_bind_host,
                "port": server_settings.server_bind_port,
                "remote_gpu_target": server_settings.remote_gpu_target,
            }
        )
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()
    return 0


def _build_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload: object, *, status_code: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(content_length) if content_length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("Expected JSON object payload.")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                self._write_json({"ok": True, "status": "healthy"})
                return
            if parsed.path == "/readyz":
                query = parse_qs(parsed.query)
                include_model_load_check = query.get("load_models", ["false"])[0].lower() in {"1", "true", "yes", "on"}
                if include_model_load_check:
                    payload, status = _warmup_payload(source="readyz_alias")
                else:
                    payload, status = _readyz_payload()
                self._write_json(payload, status_code=status)
                return
            if parsed.path == "/runtime-info":
                payload, status = _runtime_info_payload()
                self._write_json(payload, status_code=status)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/bootstrap":
                try:
                    server_settings = get_server_runtime_settings()
                    bootstrapper = WeightsBootstrapper(
                        cache_dir=Path(server_settings.model_cache_dir),
                        hf_token=server_settings.hf_token,
                    )
                    manifest = bootstrapper.load_manifest(Path(server_settings.weights_manifest_path))
                    if not manifest.artifacts:
                        raise FileNotFoundError(
                            f"No model artifacts are defined in {server_settings.weights_manifest_path}."
                        )
                    resolved = bootstrapper.ensure_manifest(manifest)
                    build_tooling_runtime.cache_clear()
                    self._write_json({"ok": True, "artifacts": {name: str(path) for name, path in resolved.items()}})
                except Exception as exc:  # noqa: BLE001
                    self._write_json({"ok": False, "error": str(exc)}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if self.path == "/warmup":
                try:
                    payload, status = _warmup_payload()
                    self._write_json(payload, status_code=status)
                except Exception as exc:  # noqa: BLE001
                    self._write_json({"ok": False, "warmup_ok": False, "error": str(exc)}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if self.path in {"/upload", "/analyze"}:
                self._write_json(
                    {
                        "ok": False,
                        "error": f"{self.path} was removed. The GPU server is inference-only; run orchestration locally and use /infer/* endpoints.",
                    },
                    status_code=HTTPStatus.GONE,
                )
                return
            if self.path == "/infer/sam3-extract":
                self._handle_sam3_extract()
                return
            if self.path == "/infer/embed-crops":
                self._handle_embed_crops()
                return
            if self.path == "/infer/score-labels":
                self._handle_score_labels()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def _handle_sam3_extract(self) -> None:
            try:
                tooling_runtime = build_tooling_runtime()
                with tempfile.TemporaryDirectory(prefix="v2a_infer_sam3_") as tmp_dir:
                    manifest_payload, file_map = _parse_multipart_form(self, output_root=Path(tmp_dir))
                    manifest = Sam3ExtractManifest.model_validate(manifest_payload)
                    frame_batches = _frame_batches_from_manifest(manifest, file_map)
                    result = tooling_runtime.sam3_client.extract_entities(
                        frame_batches,
                        prompts_by_scene=manifest.prompts_by_scene,
                        region_seeds_by_scene=manifest.region_seeds_by_scene,
                        score_threshold=manifest.score_threshold,
                        min_points=manifest.min_points,
                        high_confidence_threshold=manifest.high_confidence_threshold,
                        match_threshold=manifest.match_threshold,
                    )
                    self._write_json(result.model_dump(mode="json"))
            except RequestValidationError as exc:
                self._write_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
            except Exception as exc:  # noqa: BLE001
                self._write_json({"ok": False, "error": str(exc)}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _handle_embed_crops(self) -> None:
            try:
                tooling_runtime = build_tooling_runtime()
                with tempfile.TemporaryDirectory(prefix="v2a_infer_embed_") as tmp_dir:
                    manifest_payload, file_map = _parse_multipart_form(self, output_root=Path(tmp_dir))
                    manifest = EmbedImagesManifest.model_validate(manifest_payload)
                    image_paths_by_track = {
                        batch.track_id: [str(file_map[image.upload_key]) for image in batch.images]
                        for batch in manifest.tracks
                    }
                    result = tooling_runtime.embedding_client.embed_images(image_paths_by_track)
                    self._write_json([item.model_dump(mode="json") for item in result])
            except RequestValidationError as exc:
                self._write_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
            except Exception as exc:  # noqa: BLE001
                self._write_json({"ok": False, "error": str(exc)}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

        def _handle_score_labels(self) -> None:
            try:
                tooling_runtime = build_tooling_runtime()
                with tempfile.TemporaryDirectory(prefix="v2a_infer_labels_") as tmp_dir:
                    manifest_payload, file_map = _parse_multipart_form(self, output_root=Path(tmp_dir))
                    manifest = LabelScoreManifest.model_validate(manifest_payload)
                    image_paths = [str(file_map[image.upload_key]) for image in manifest.images]
                    result = tooling_runtime.label_client.score_image_labels(
                        image_paths=image_paths,
                        labels=list(manifest.labels),
                    )
                    self._write_json([item.model_dump(mode="json") for item in result])
            except RequestValidationError as exc:
                self._write_json({"ok": False, "error": str(exc)}, status_code=exc.status_code)
            except Exception as exc:  # noqa: BLE001
                self._write_json({"ok": False, "error": str(exc)}, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return None

    return Handler


def _parse_multipart_form(
    handler: BaseHTTPRequestHandler,
    *,
    output_root: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise RequestValidationError("Expected multipart/form-data request.", HTTPStatus.BAD_REQUEST)
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        raise RequestValidationError(
            "multipart request must include Content-Length",
            HTTPStatus.LENGTH_REQUIRED,
        )
    if content_length > _MAX_MULTIPART_REQUEST_BYTES:
        raise RequestValidationError(
            f"multipart request exceeds {_MAX_MULTIPART_REQUEST_BYTES} bytes",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )
    environ = {
        "REQUEST_METHOD": "POST",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(content_length),
    }
    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ=environ,
        keep_blank_values=True,
    )
    manifest_raw = form.getvalue("manifest")
    if not isinstance(manifest_raw, str):
        raise RequestValidationError(
            "multipart request must include a JSON manifest field",
            HTTPStatus.BAD_REQUEST,
        )
    manifest_payload = json.loads(manifest_raw)
    if not isinstance(manifest_payload, dict):
        raise RequestValidationError(
            "manifest must decode to a JSON object",
            HTTPStatus.BAD_REQUEST,
        )
    file_map: dict[str, Path] = {}
    output_root_resolved = output_root.resolve()
    for key in form.keys():
        if key == "manifest":
            continue
        field = form[key]
        if isinstance(field, list):
            raise RequestValidationError(
                f"Duplicate multipart field: {key}",
                HTTPStatus.BAD_REQUEST,
            )
        if len(file_map) >= _MAX_MULTIPART_FILE_COUNT:
            raise RequestValidationError(
                f"multipart request exceeds {_MAX_MULTIPART_FILE_COUNT} files",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        safe_key = _sanitize_multipart_token(key, label="multipart field name")
        filename = Path(field.filename or key).name
        safe_filename = _sanitize_multipart_token(
            filename or safe_key,
            label="multipart filename",
        )
        target = (output_root / f"{safe_key}-{safe_filename}").resolve()
        try:
            target.relative_to(output_root_resolved)
        except ValueError as exc:
            raise RequestValidationError(
                "multipart field escapes output directory",
                HTTPStatus.BAD_REQUEST,
            ) from exc
        payload = field.file.read(_MAX_MULTIPART_FILE_BYTES + 1)
        if len(payload) > _MAX_MULTIPART_FILE_BYTES:
            raise RequestValidationError(
                f"multipart file exceeds {_MAX_MULTIPART_FILE_BYTES} bytes",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        with target.open("wb") as file_obj:
            file_obj.write(payload)
        file_map[key] = target
    return manifest_payload, file_map


def _frame_batches_from_manifest(
    manifest: Sam3ExtractManifest,
    file_map: dict[str, Path],
) -> list[FrameBatch]:
    frame_batches: list[FrameBatch] = []
    for batch in manifest.frame_batches:
        frames = [
            SampledFrame(
                scene_index=frame.scene_index,
                timestamp_seconds=frame.timestamp_seconds,
                image_path=str(file_map[frame.upload_key]),
            )
            for frame in batch.frames
        ]
        frame_batches.append(FrameBatch(scene_index=batch.scene_index, frames=frames))
    return frame_batches


class RequestValidationError(ValueError):
    def __init__(self, message: str, status_code: HTTPStatus) -> None:
        super().__init__(message)
        self.status_code = status_code


def _sanitize_multipart_token(token: str, *, label: str) -> str:
    normalized = token.strip()
    if not normalized or not _SAFE_MULTIPART_TOKEN.fullmatch(normalized):
        raise RequestValidationError(
            f"Invalid {label}: {token!r}",
            HTTPStatus.BAD_REQUEST,
        )
    return normalized
