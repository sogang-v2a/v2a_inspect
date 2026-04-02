from __future__ import annotations

from typing import Any

from v2a_inspect.tools import RemoteGpuPolicy, RemoteGpuSelection, choose_remote_gpu

from ..remote import get_json, post_json
from .base import (
    GpuProvider,
    ProviderJobRef,
    ProviderJobState,
    ProviderJobStatus,
    ProviderResult,
    ProviderServiceConfig,
)


class RunpodProvider(GpuProvider):
    provider_name = "runpod"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        gpu_policy: RemoteGpuPolicy,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.gpu_policy = gpu_policy
        self.timeout_seconds = timeout_seconds

    def select_gpu(self, policy: RemoteGpuPolicy) -> RemoteGpuSelection:
        return choose_remote_gpu(policy)

    def invoke(
        self,
        service: ProviderServiceConfig,
        payload: dict[str, Any],
    ) -> ProviderResult:
        response = post_json(
            f"{self.base_url}/{service.route.lstrip('/')}",
            {
                "gpu": self.select_gpu(self.gpu_policy).model_dump(mode="json"),
                "input": payload,
            },
            api_key=self.api_key,
            timeout_seconds=service.timeout_seconds or self.timeout_seconds,
        )
        return ProviderResult(
            provider=self.provider_name,
            service=service.name,
            payload=response.get("output", response),
        )

    def submit_job(
        self,
        service: ProviderServiceConfig,
        payload: dict[str, Any],
    ) -> ProviderJobRef:
        response = post_json(
            f"{self.base_url}/{service.route.lstrip('/')}",
            {
                "gpu": self.select_gpu(self.gpu_policy).model_dump(mode="json"),
                "input": payload,
            },
            api_key=self.api_key,
            timeout_seconds=service.timeout_seconds or self.timeout_seconds,
        )
        job_id = str(
            response.get("id")
            or response.get("job_id")
            or response.get("request_id")
            or ""
        )
        if not job_id:
            raise ValueError("Runpod async submission did not return a job id.")
        return ProviderJobRef(
            provider=self.provider_name,
            service=service.name,
            job_id=job_id,
        )

    def poll_job(self, job_ref: ProviderJobRef) -> ProviderJobStatus:
        payload = get_json(
            f"{self.base_url}/jobs/{job_ref.job_id}",
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        state = str(payload.get("state") or payload.get("status") or "queued")
        normalized_state: ProviderJobState
        if state in {"queued", "pending"}:
            normalized_state = "queued"
        elif state in {"running", "processing", "in_progress"}:
            normalized_state = "running"
        elif state in {"completed", "success", "succeeded"}:
            normalized_state = "completed"
        else:
            normalized_state = "failed"
        return ProviderJobStatus(
            provider=self.provider_name,
            service=job_ref.service,
            job_id=job_ref.job_id,
            state=normalized_state,
            payload=payload,
        )

    def fetch_result(self, job_ref: ProviderJobRef) -> ProviderResult:
        payload = get_json(
            f"{self.base_url}/jobs/{job_ref.job_id}",
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        return ProviderResult(
            provider=self.provider_name,
            service=job_ref.service,
            payload=payload.get("output", payload),
            job_ref=job_ref,
        )
