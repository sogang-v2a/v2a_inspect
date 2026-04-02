from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    memory_total_mib: int


@dataclass(frozen=True)
class RuntimeCheckResult:
    available: bool
    devices: list[GpuDevice]
    minimum_vram_gb: int
    message: str


def inspect_nvidia_runtime(*, minimum_vram_gb: int) -> RuntimeCheckResult:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return RuntimeCheckResult(
            available=False,
            devices=[],
            minimum_vram_gb=minimum_vram_gb,
            message="nvidia-smi is not available in this environment.",
        )

    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = _parse_gpu_rows(completed.stdout)
    minimum_mib = minimum_vram_gb * 1024
    has_eligible_gpu = any(device.memory_total_mib >= minimum_mib for device in devices)
    if not devices:
        return RuntimeCheckResult(
            available=False,
            devices=[],
            minimum_vram_gb=minimum_vram_gb,
            message="No NVIDIA GPUs were detected by nvidia-smi.",
        )
    if not has_eligible_gpu:
        return RuntimeCheckResult(
            available=False,
            devices=devices,
            minimum_vram_gb=minimum_vram_gb,
            message="Detected GPUs do not meet the minimum VRAM requirement.",
        )
    return RuntimeCheckResult(
        available=True,
        devices=devices,
        minimum_vram_gb=minimum_vram_gb,
        message="NVIDIA runtime check passed.",
    )


def runtime_check_to_json(result: RuntimeCheckResult) -> str:
    return json.dumps(
        {
            "available": result.available,
            "minimum_vram_gb": result.minimum_vram_gb,
            "message": result.message,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "memory_total_mib": device.memory_total_mib,
                }
                for device in result.devices
            ],
        },
        indent=2,
    )


def _parse_gpu_rows(payload: str) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    for row in payload.splitlines():
        if not row.strip():
            continue
        index_text, name, memory_text = [part.strip() for part in row.split(",", 2)]
        devices.append(
            GpuDevice(
                index=int(index_text),
                name=name,
                memory_total_mib=int(memory_text),
            )
        )
    return devices
