from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from v2a_inspect.settings import Settings, settings


@dataclass(frozen=True)
class ClientRuntimeSettings:
    server_base_url: str
    shared_video_dir: Path | None
    remote_timeout_seconds: int
    ui_analysis_concurrency_limit: int
    ui_analysis_acquire_timeout_seconds: int
    ui_temp_cleanup_max_age_seconds: int
    ui_cleanup_interval_seconds: int


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


def get_client_runtime_settings(
    resolved_settings: Settings | None = None,
) -> ClientRuntimeSettings:
    base = resolved_settings or settings
    return ClientRuntimeSettings(
        server_base_url=base.server_base_url,
        shared_video_dir=base.shared_video_dir,
        remote_timeout_seconds=base.remote_timeout_seconds,
        ui_analysis_concurrency_limit=base.ui_analysis_concurrency_limit,
        ui_analysis_acquire_timeout_seconds=base.ui_analysis_acquire_timeout_seconds,
        ui_temp_cleanup_max_age_seconds=base.ui_temp_cleanup_max_age_seconds,
        ui_cleanup_interval_seconds=base.ui_cleanup_interval_seconds,
    )


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
    )


def required_env_vars_by_runtime_mode() -> dict[str, list[str]]:
    return {
        "client_ui": [
            "SERVER_BASE_URL",
            "SHARED_VIDEO_DIR",
            "REMOTE_TIMEOUT_SECONDS",
            "UI_ANALYSIS_CONCURRENCY_LIMIT",
            "UI_ANALYSIS_ACQUIRE_TIMEOUT_SECONDS",
        ],
        "server_runtime": [
            "HF_TOKEN",
            "MODEL_CACHE_DIR",
            "WEIGHTS_MANIFEST_PATH",
            "RUNTIME_PROFILE",
            "REMOTE_GPU_TARGET",
            "MINIMUM_GPU_VRAM_GB",
            "SERVER_BIND_HOST",
            "SERVER_BIND_PORT",
            "SHARED_VIDEO_DIR",
        ],
    }
