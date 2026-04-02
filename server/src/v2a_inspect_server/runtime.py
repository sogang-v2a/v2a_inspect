from __future__ import annotations

from dataclasses import dataclass

from v2a_inspect.settings import settings
from v2a_inspect.tools import RemoteGpuPolicy, RemoteGpuSelection, choose_remote_gpu

from .embeddings import EmbeddingRunpodClient, Siglip2LabelClient
from .sam3 import Sam3RunpodClient


@dataclass(frozen=True)
class ToolingRuntime:
    gpu_selection: RemoteGpuSelection
    sam3_client: Sam3RunpodClient | None
    embedding_client: EmbeddingRunpodClient | None
    label_client: Siglip2LabelClient | None


def build_tooling_runtime() -> ToolingRuntime:
    gpu_policy = RemoteGpuPolicy(
        preferred_sku=settings.remote_gpu_preference,
        fallback_sku=settings.remote_gpu_fallback,
        preferred_vram_gb=settings.remote_gpu_vram_preference_gb,
        max_vram_gb=settings.remote_gpu_vram_cap_gb,
    )
    api_key = (
        settings.runpod_api_key.get_secret_value()
        if settings.runpod_api_key is not None
        else None
    )
    timeout = settings.remote_timeout_seconds
    return ToolingRuntime(
        gpu_selection=choose_remote_gpu(gpu_policy),
        sam3_client=(
            Sam3RunpodClient(
                endpoint_url=settings.sam3_endpoint_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )
            if settings.sam3_endpoint_url
            else None
        ),
        embedding_client=(
            EmbeddingRunpodClient(
                endpoint_url=settings.embedding_endpoint_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )
            if settings.embedding_endpoint_url
            else None
        ),
        label_client=(
            Siglip2LabelClient(
                endpoint_url=settings.label_endpoint_url,
                api_key=api_key,
                timeout_seconds=timeout,
            )
            if settings.label_endpoint_url
            else None
        ),
    )
