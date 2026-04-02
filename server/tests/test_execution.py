from __future__ import annotations

import unittest

from v2a_inspect.tools import RemoteGpuPolicy, choose_remote_gpu
from v2a_inspect_server.execution import execute_service
from v2a_inspect_server.providers import (
    GpuProvider,
    ProviderJobRef,
    ProviderJobStatus,
    ProviderResult,
    ProviderServiceConfig,
)


class _FakeAsyncProvider(GpuProvider):
    provider_name = "fake"

    def __init__(self) -> None:
        self.poll_count = 0

    def select_gpu(self, policy: RemoteGpuPolicy):
        return choose_remote_gpu(policy)

    def invoke(
        self,
        service: ProviderServiceConfig,
        payload: dict[str, object],
    ) -> ProviderResult:
        return ProviderResult(
            provider="fake", service=service.name, payload={"ok": True}
        )

    def submit_job(
        self,
        service: ProviderServiceConfig,
        payload: dict[str, object],
    ) -> ProviderJobRef:
        return ProviderJobRef(provider="fake", service=service.name, job_id="job-1")

    def poll_job(self, job_ref: ProviderJobRef) -> ProviderJobStatus:
        self.poll_count += 1
        state = "completed" if self.poll_count > 1 else "running"
        return ProviderJobStatus(
            provider="fake",
            service=job_ref.service,
            job_id=job_ref.job_id,
            state=state,
            payload={},
        )

    def fetch_result(self, job_ref: ProviderJobRef) -> ProviderResult:
        return ProviderResult(
            provider="fake",
            service=job_ref.service,
            payload={"done": True},
            job_ref=job_ref,
        )


class ExecutionTests(unittest.TestCase):
    def test_execute_service_sync(self) -> None:
        provider = _FakeAsyncProvider()
        result = execute_service(
            provider,
            ProviderServiceConfig(name="sam3", route="sam3", mode="sync_endpoint"),
            {"hello": "world"},
            poll_interval_seconds=0.0,
        )
        self.assertEqual(result.payload, {"ok": True})

    def test_execute_service_async(self) -> None:
        provider = _FakeAsyncProvider()
        result = execute_service(
            provider,
            ProviderServiceConfig(name="sam3", route="sam3", mode="async_job"),
            {"hello": "world"},
            poll_interval_seconds=0.0,
            max_polls=3,
        )
        self.assertEqual(result.payload, {"done": True})


if __name__ == "__main__":
    unittest.main()
