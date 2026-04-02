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
    mode = "sync_endpoint"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def select_gpu(self, policy: RemoteGpuPolicy) -> RemoteGpuSelection:
        return choose_remote_gpu(policy)

    def invoke(
        self,
        *,
        service: ProviderServiceConfig,
        payload: dict[str, Any],
        gpu_policy: RemoteGpuPolicy,
    ) -> ProviderResult:
        response = post_json(
            f"{self.base_url}/{service.resolved_route.lstrip('/')}",
            _build_request_payload(self.select_gpu(gpu_policy), payload),
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        return ProviderResult(
            provider=self.provider_name,
            service=service.name,
            payload=_extract_output_payload(response),
        )

    def submit_job(
        self,
        *,
        service: ProviderServiceConfig,
        payload: dict[str, Any],
        gpu_policy: RemoteGpuPolicy,
    ) -> ProviderJobRef:
        response = post_json(
            f"{self.base_url}/jobs/{service.resolved_route.lstrip('/')}",
            _build_request_payload(self.select_gpu(gpu_policy), payload),
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        job_id = response.get("job_id", response.get("id"))
        if not isinstance(job_id, str) or not job_id:
            raise TypeError("Runpod provider did not return a valid job_id.")
        return ProviderJobRef(
            provider=self.provider_name, service=service.name, job_id=job_id
        )

    def poll_job(self, job_ref: ProviderJobRef) -> ProviderJobStatus:
        response = get_json(
            f"{self.base_url}/jobs/{job_ref.job_id}",
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        raw_state = str(response.get("state") or response.get("status") or "queued")
        normalized_state: ProviderJobState
        if raw_state in {"queued", "pending"}:
            normalized_state = "queued"
        elif raw_state in {"running", "processing", "in_progress"}:
            normalized_state = "running"
        elif raw_state in {"completed", "success", "succeeded"}:
            normalized_state = "completed"
        else:
            normalized_state = "failed"
        return ProviderJobStatus(
            provider=self.provider_name,
            service=job_ref.service,
            job_id=job_ref.job_id,
            state=normalized_state,
            payload=response,
        )

    def fetch_result(self, job_ref: ProviderJobRef) -> ProviderResult:
        response = get_json(
            f"{self.base_url}/jobs/{job_ref.job_id}/result",
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        return ProviderResult(
            provider=self.provider_name,
            service=job_ref.service,
            payload=_extract_output_payload(response),
            job_ref=job_ref,
        )


def _extract_output_payload(response: dict[str, Any]) -> dict[str, Any]:
    payload = response.get("output", response)
    if isinstance(payload, dict):
        return payload
    raise TypeError("Provider response payload must be a JSON object.")


def _build_request_payload(
    gpu_selection: RemoteGpuSelection,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "gpu": gpu_selection.model_dump(mode="json"),
        "input": payload,
    }
