from __future__ import annotations

import json
import re
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
    inventory = subprocess.run(
        [nvidia_smi, "-L"],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = _parse_gpu_rows(completed.stdout, inventory_payload=inventory.stdout)
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


def _parse_gpu_rows(payload: str, *, inventory_payload: str) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    pending_fallbacks: list[tuple[int, str]] = []
    for row in payload.splitlines():
        if not row.strip():
            continue
        index_text, name, memory_text = [part.strip() for part in row.split(",", 2)]
        try:
            memory_total_mib = int(memory_text)
        except ValueError:
            pending_fallbacks.append((int(index_text), name))
            continue
        devices.append(
            GpuDevice(
                index=int(index_text),
                name=name,
                memory_total_mib=memory_total_mib,
            )
        )
    if pending_fallbacks:
        devices.extend(_fallback_devices_from_inventory(pending_fallbacks, inventory_payload))
    return devices


def _fallback_devices_from_inventory(
    pending_fallbacks: list[tuple[int, str]],
    inventory_payload: str,
) -> list[GpuDevice]:
    mig_devices = _parse_mig_inventory(inventory_payload)
    if mig_devices:
        return mig_devices
    devices: list[GpuDevice] = []
    for index, name in pending_fallbacks:
        devices.append(
            GpuDevice(
                index=index,
                name=name,
                memory_total_mib=_memory_from_name(name),
            )
        )
    return devices


def _parse_mig_inventory(payload: str) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    for row in payload.splitlines():
        match = re.search(r"MIG\s+[^.]+\.(\d+)gb\s+Device\s+(\d+)", row, re.IGNORECASE)
        if match is None:
            continue
        memory_gb = int(match.group(1))
        device_index = int(match.group(2))
        devices.append(
            GpuDevice(
                index=device_index,
                name=row.strip().split("(", 1)[0].strip(),
                memory_total_mib=memory_gb * 1024,
            )
        )
    return devices


def _memory_from_name(name: str) -> int:
    match = re.search(r"(\d+)GB", name, re.IGNORECASE)
    if match is None:
        return 0
    return int(match.group(1)) * 1024
