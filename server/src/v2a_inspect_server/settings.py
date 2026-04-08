from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from v2a_inspect.settings import Settings, settings


@dataclass(frozen=True)
class ServerRuntimeSettings:
    runtime_mode: Literal["nvidia_docker", "in_process"]
    server_bind_host: str
    server_bind_port: int
    shared_video_dir: Path | None
    minimum_gpu_vram_gb: int
    model_cache_dir: Path
    weights_manifest_path: Path
    hf_token: str | None


def get_server_runtime_settings(
    resolved_settings: Settings | None = None,
) -> ServerRuntimeSettings:
    base = resolved_settings or settings
    return ServerRuntimeSettings(
        runtime_mode=base.runtime_mode,
        server_bind_host=base.server_bind_host,
        server_bind_port=base.server_bind_port,
        shared_video_dir=base.shared_video_dir,
        minimum_gpu_vram_gb=base.minimum_gpu_vram_gb,
        model_cache_dir=base.model_cache_dir,
        weights_manifest_path=base.weights_manifest_path,
        hf_token=(
            base.hf_token.get_secret_value() if base.hf_token is not None else None
        ),
    )
