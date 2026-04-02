from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


GpuSku = Literal["A4000", "A4500"]


class RemoteGpuPolicy(BaseModel):
    preferred_sku: GpuSku = "A4000"
    fallback_sku: GpuSku = "A4500"
    preferred_vram_gb: int = Field(default=16, ge=1, le=24)
    max_vram_gb: int = Field(default=24, ge=1, le=24)


class RemoteGpuSelection(BaseModel):
    sku: GpuSku
    vram_gb: int = Field(ge=1, le=24)
    source: Literal["preferred", "fallback"]


def choose_remote_gpu(policy: RemoteGpuPolicy) -> RemoteGpuSelection:
    if policy.preferred_vram_gb > policy.max_vram_gb:
        raise ValueError("Preferred VRAM cannot exceed the maximum VRAM cap.")

    if policy.preferred_sku == "A4000" and policy.preferred_vram_gb <= 16:
        return RemoteGpuSelection(
            sku="A4000",
            vram_gb=16,
            source="preferred",
        )

    if policy.preferred_sku == "A4500" and policy.max_vram_gb >= 20:
        return RemoteGpuSelection(
            sku="A4500",
            vram_gb=min(24, policy.max_vram_gb),
            source="preferred",
        )

    if policy.fallback_sku == "A4000":
        return RemoteGpuSelection(
            sku="A4000",
            vram_gb=16,
            source="fallback",
        )

    return RemoteGpuSelection(
        sku="A4500",
        vram_gb=min(24, max(20, policy.max_vram_gb)),
        source="fallback",
    )
