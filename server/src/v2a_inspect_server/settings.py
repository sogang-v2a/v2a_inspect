from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from v2a_inspect.settings import Settings, settings


@dataclass(frozen=True)
class ServerRuntimeSettings:
    runtime_mode: Literal["nvidia_docker", "in_process"]
    runtime_profile: Literal["mig10_safe", "full_gpu", "cpu_dev"]
    remote_gpu_target: str
    server_bind_host: str
    server_bind_port: int
    shared_video_dir: Path | None
    minimum_gpu_vram_gb: int
    model_cache_dir: Path
    weights_manifest_path: Path
    hf_token: str | None
    gemini_api_key: str | None


def get_server_runtime_settings(
    resolved_settings: Settings | None = None,
) -> ServerRuntimeSettings:
    base = resolved_settings or settings
    return ServerRuntimeSettings(
        runtime_mode=base.runtime_mode,
        runtime_profile=base.runtime_profile,
        remote_gpu_target=base.remote_gpu_target,
        server_bind_host=base.server_bind_host,
        server_bind_port=base.server_bind_port,
        shared_video_dir=base.shared_video_dir,
        minimum_gpu_vram_gb=base.minimum_gpu_vram_gb,
        model_cache_dir=base.model_cache_dir,
        weights_manifest_path=base.weights_manifest_path,
        hf_token=(
            base.hf_token.get_secret_value() if base.hf_token is not None else None
        ),
        gemini_api_key=(
            base.gemini_api_key.get_secret_value()
            if base.gemini_api_key is not None
            else None
        ),
    )
