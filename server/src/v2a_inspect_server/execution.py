from __future__ import annotations

import time
from collections.abc import Mapping

from v2a_inspect.tools import RemoteGpuPolicy

from .providers import GpuProvider, ProviderResult, ProviderServiceConfig


def execute_service(
    provider: GpuProvider,
    service: ProviderServiceConfig,
    payload: Mapping[str, object],
    *,
    gpu_policy: RemoteGpuPolicy,
    poll_interval_seconds: float = 0.1,
    max_polls: int = 120,
) -> ProviderResult:
    if service.mode == "sync_endpoint":
        return provider.invoke(
            service=service,
            payload=dict(payload),
            gpu_policy=gpu_policy,
        )

    job_ref = provider.submit_job(
        service=service,
        payload=dict(payload),
        gpu_policy=gpu_policy,
    )
    for _ in range(max_polls):
        status = provider.poll_job(job_ref)
        if status.state == "completed":
            return provider.fetch_result(job_ref)
        if status.state == "failed":
            raise RuntimeError(
                f"Provider job failed for service {service.name!r}: {status.payload}"
            )
        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out waiting for provider job {job_ref.job_id!r} "
        f"for service {service.name!r}."
    )
