from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

from v2a_inspect.settings import settings
from v2a_inspect.tools import RemoteGpuPolicy, RemoteGpuSelection, choose_remote_gpu

from .bootstrap import WeightsBootstrapper, WeightsManifest
from .embeddings import EmbeddingClient, LabelClient
from .gpu_runtime import inspect_nvidia_runtime, runtime_check_to_json
from .providers import GpuProvider, RunpodProvider
from .sam3 import Sam3Client


@dataclass(frozen=True)
class ToolingRuntime:
    gpu_selection: RemoteGpuSelection
    provider: GpuProvider
    sam3_client: Sam3Client
    embedding_client: EmbeddingClient
    label_client: LabelClient
    bootstrapper: WeightsBootstrapper
    weights_manifest: WeightsManifest


def build_tooling_runtime() -> ToolingRuntime:
    gpu_policy = RemoteGpuPolicy(
        preferred_sku=settings.remote_gpu_preference,
        fallback_sku=settings.remote_gpu_fallback,
        preferred_vram_gb=settings.remote_gpu_vram_preference_gb,
        max_vram_gb=settings.remote_gpu_vram_cap_gb,
    )
    provider = _build_provider()
    bootstrapper = WeightsBootstrapper(
        cache_dir=Path(settings.model_cache_dir),
        hf_token=(
            settings.hf_token.get_secret_value()
            if settings.hf_token is not None
            else None
        ),
    )
    weights_manifest = bootstrapper.load_manifest(Path(settings.weights_manifest_path))
    return ToolingRuntime(
        gpu_selection=choose_remote_gpu(gpu_policy),
        provider=provider,
        sam3_client=Sam3Client(
            provider=provider,
            service=settings.sam3_service,
            gpu_policy=gpu_policy,
            mode=settings.provider_mode,
        ),
        embedding_client=EmbeddingClient(
            provider=provider,
            service=settings.embedding_service,
            gpu_policy=gpu_policy,
            mode=settings.provider_mode,
        ),
        label_client=LabelClient(
            provider=provider,
            service=settings.label_service,
            gpu_policy=gpu_policy,
            mode=settings.provider_mode,
        ),
        bootstrapper=bootstrapper,
        weights_manifest=weights_manifest,
    )


def _build_provider() -> GpuProvider:
    api_key = (
        settings.gpu_provider_api_key.get_secret_value()
        if settings.gpu_provider_api_key is not None
        else None
    )
    if settings.gpu_provider != "runpod":
        raise ValueError(
            f"Unsupported GPU provider {settings.gpu_provider!r}. Only 'runpod' is implemented in this slice."
        )
    if not settings.provider_base_url:
        raise ValueError(
            "GPU_PROVIDER_BASE_URL / RUNPOD_BASE_URL must be set for the tooling runtime."
        )

    return RunpodProvider(
        base_url=settings.provider_base_url,
        api_key=api_key,
        timeout_seconds=settings.remote_timeout_seconds,
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

    args = parser.parse_args(argv)
    if args.command == "runtime-info":
        return _run_runtime_info()
    if args.command == "bootstrap":
        return _run_bootstrap()
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
    print(
        json.dumps(
            {name: str(path) for name, path in resolved.items()},
            indent=2,
        )
    )
    return 0


def _run_check() -> int:
    result = inspect_nvidia_runtime(minimum_vram_gb=settings.minimum_gpu_vram_gb)
    print(runtime_check_to_json(result))
    return 0 if result.available else 1
