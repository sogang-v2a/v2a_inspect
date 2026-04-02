from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from v2a_inspect.tools import RemoteGpuPolicy, RemoteGpuSelection

ProviderMode = Literal["sync_endpoint", "async_job"]
ProviderJobState = Literal["queued", "running", "completed", "failed"]


class ProviderServiceConfig(BaseModel):
    name: str
    route: str
    mode: ProviderMode = "sync_endpoint"
    timeout_seconds: int = Field(default=120, ge=1)

    @property
    def resolved_route(self) -> str:
        return self.route or self.name


class ProviderJobRef(BaseModel):
    provider: str
    service: str
    job_id: str


class ProviderJobStatus(BaseModel):
    provider: str
    service: str
    job_id: str
    state: ProviderJobState
    payload: dict[str, Any] = Field(default_factory=dict)


class ProviderResult(BaseModel):
    provider: str
    service: str
    payload: dict[str, Any] = Field(default_factory=dict)
    job_ref: ProviderJobRef | None = None


class GpuProvider(ABC):
    provider_name: str

    @abstractmethod
    def select_gpu(self, policy: RemoteGpuPolicy) -> RemoteGpuSelection:
        raise NotImplementedError

    @abstractmethod
    def invoke(
        self,
        *,
        service: ProviderServiceConfig,
        payload: dict[str, Any],
        gpu_policy: RemoteGpuPolicy,
    ) -> ProviderResult:
        raise NotImplementedError

    @abstractmethod
    def submit_job(
        self,
        *,
        service: ProviderServiceConfig,
        payload: dict[str, Any],
        gpu_policy: RemoteGpuPolicy,
    ) -> ProviderJobRef:
        raise NotImplementedError

    @abstractmethod
    def poll_job(self, job_ref: ProviderJobRef) -> ProviderJobStatus:
        raise NotImplementedError

    @abstractmethod
    def fetch_result(self, job_ref: ProviderJobRef) -> ProviderResult:
        raise NotImplementedError
