from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from v2a_inspect.settings import settings
from v2a_inspect.tools import RemoteGpuPolicy, RemoteGpuSelection, choose_remote_gpu

from .bootstrap import WeightsBootstrapper, WeightsManifest
from .embeddings import EmbeddingClient, LabelClient
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
