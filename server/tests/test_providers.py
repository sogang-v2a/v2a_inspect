from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from v2a_inspect.tools import RemoteGpuPolicy
from v2a_inspect_server.providers import ProviderServiceConfig, RunpodProvider


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class RunpodProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gpu_policy = RemoteGpuPolicy(
            preferred_sku="A4000",
            fallback_sku="A4500",
            preferred_vram_gb=16,
            max_vram_gb=24,
        )

    @patch("v2a_inspect_server.remote.request.urlopen")
    def test_invoke_returns_provider_result(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse({"output": {"ok": True}})
        provider = RunpodProvider(base_url="https://gpu.example.com", api_key="secret")
        result = provider.invoke(
            service=ProviderServiceConfig(name="sam3", route="sam3/run"),
            payload={"hello": "world"},
            gpu_policy=self.gpu_policy,
        )
        self.assertEqual(result.provider, "runpod")
        self.assertEqual(result.service, "sam3")
        self.assertEqual(result.payload, {"ok": True})

    @patch("v2a_inspect_server.remote.request.urlopen")
    def test_submit_job_creates_job_ref(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse({"id": "job-123"})
        provider = RunpodProvider(base_url="https://gpu.example.com", api_key="secret")
        job_ref = provider.submit_job(
            service=ProviderServiceConfig(
                name="sam3", route="sam3/jobs", mode="async_job"
            ),
            payload={"hello": "world"},
            gpu_policy=self.gpu_policy,
        )
        self.assertEqual(job_ref.provider, "runpod")
        self.assertEqual(job_ref.job_id, "job-123")


if __name__ == "__main__":
    unittest.main()
