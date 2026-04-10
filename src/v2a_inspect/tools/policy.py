from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RuntimeProfile = Literal["mig10_safe", "full_gpu"]


class RemoteGpuPolicy(BaseModel):
    target: str = "sogang_gpu"
    preferred_profile: RuntimeProfile = "mig10_safe"
    fallback_profile: RuntimeProfile = "full_gpu"
    preferred_vram_gb: int = Field(default=10, ge=1, le=80)
    max_vram_gb: int = Field(default=80, ge=1, le=80)


class RemoteGpuSelection(BaseModel):
    target: str
    runtime_profile: RuntimeProfile
    vram_gb: int = Field(ge=1, le=80)
    source: Literal["preferred", "fallback"]


def choose_remote_gpu(policy: RemoteGpuPolicy) -> RemoteGpuSelection:
    if policy.preferred_vram_gb > policy.max_vram_gb:
        raise ValueError("Preferred VRAM cannot exceed the maximum VRAM cap.")

    if policy.preferred_profile == "mig10_safe" and policy.preferred_vram_gb <= 10:
        return RemoteGpuSelection(
            target=policy.target,
            runtime_profile="mig10_safe",
            vram_gb=10,
            source="preferred",
        )

    if policy.preferred_profile == "full_gpu":
        return RemoteGpuSelection(
            target=policy.target,
            runtime_profile="full_gpu",
            vram_gb=min(policy.max_vram_gb, policy.preferred_vram_gb),
            source="preferred",
        )

    if policy.fallback_profile == "mig10_safe":
        return RemoteGpuSelection(
            target=policy.target,
            runtime_profile="mig10_safe",
            vram_gb=10,
            source="fallback",
        )

    return RemoteGpuSelection(
        target=policy.target,
        runtime_profile="full_gpu",
        vram_gb=min(policy.max_vram_gb, policy.preferred_vram_gb),
        source="fallback",
    )
