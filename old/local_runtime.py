from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from v2a_inspect.clients.inference import (
    RemoteEmbeddingClient,
    RemoteLabelClient,
    RemoteSam3Client,
    server_runtime_info,
)
from v2a_inspect.settings import settings

from .adjudicator import GeminiIssueJudge
from .description_writer import GeminiDescriptionWriter
from .gemini_grouping import GeminiGroupingJudge
from .gemini_proposal_grounding import GeminiProposalGrounder
from .gemini_routing import GeminiRoutingJudge
from .gemini_source_proposal import GeminiSourceProposer
from .gemini_source_semantics import GeminiSourceSemanticsInterpreter


@dataclass(frozen=True)
class RemoteRuntimeSnapshot:
    effective_runtime_profile: str | None
    runtime_profile_source: str | None
    residency_mode: str | None
    resident_models: list[str]


class LocalToolingRuntime:
    def __init__(
        self,
        *,
        server_base_url: str,
        remote_timeout_seconds: float,
        semantic_model: str,
    ) -> None:
        self.server_base_url = server_base_url.rstrip("/")
        self.remote_timeout_seconds = remote_timeout_seconds
        self.semantic_model = semantic_model
        self._sam3_client: RemoteSam3Client | None = None
        self._embedding_client: RemoteEmbeddingClient | None = None
        self._label_client: RemoteLabelClient | None = None
        self._description_writer: GeminiDescriptionWriter | None = None
        self._adjudication_judge: GeminiIssueJudge | None = None
        self._source_proposer: GeminiSourceProposer | None = None
        self._proposal_grounder: GeminiProposalGrounder | None = None
        self._source_semantics_interpreter: GeminiSourceSemanticsInterpreter | None = None
        self._grouping_judge: GeminiGroupingJudge | None = None
        self._routing_judge: GeminiRoutingJudge | None = None

    @property
    def should_release_clients(self) -> bool:
        return False

    @property
    def residency_mode(self) -> Literal["remote_inference"]:
        return "remote_inference"

    @property
    def sam3_client(self) -> RemoteSam3Client:
        if self._sam3_client is None:
            self._sam3_client = RemoteSam3Client(
                server_base_url=self.server_base_url,
                timeout_seconds=self.remote_timeout_seconds,
            )
        return self._sam3_client

    @property
    def embedding_client(self) -> RemoteEmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = RemoteEmbeddingClient(
                server_base_url=self.server_base_url,
                timeout_seconds=self.remote_timeout_seconds,
            )
        return self._embedding_client

    @property
    def label_client(self) -> RemoteLabelClient:
        if self._label_client is None:
            self._label_client = RemoteLabelClient(
                server_base_url=self.server_base_url,
                timeout_seconds=self.remote_timeout_seconds,
            )
        return self._label_client

    @property
    def description_writer(self) -> GeminiDescriptionWriter:
        if self._description_writer is None:
            self._description_writer = GeminiDescriptionWriter(
                model=self.semantic_model,
                api_key=_gemini_api_key_or_empty(),
            )
        return self._description_writer

    @property
    def adjudication_judge(self) -> GeminiIssueJudge:
        if self._adjudication_judge is None:
            self._adjudication_judge = GeminiIssueJudge(
                model=self.semantic_model,
                api_key=_gemini_api_key_or_empty(),
            )
        return self._adjudication_judge

    @property
    def source_proposer(self) -> GeminiSourceProposer:
        if self._source_proposer is None:
            self._source_proposer = GeminiSourceProposer(
                model=self.semantic_model,
                api_key=_gemini_api_key_or_empty(),
            )
        return self._source_proposer

    @property
    def proposal_grounder(self) -> GeminiProposalGrounder:
        if self._proposal_grounder is None:
            self._proposal_grounder = GeminiProposalGrounder(
                model=self.semantic_model,
                api_key=_gemini_api_key_or_empty(),
            )
        return self._proposal_grounder

    @property
    def source_semantics_interpreter(self) -> GeminiSourceSemanticsInterpreter:
        if self._source_semantics_interpreter is None:
            self._source_semantics_interpreter = GeminiSourceSemanticsInterpreter(
                model=self.semantic_model,
                api_key=_gemini_api_key_or_empty(),
            )
        return self._source_semantics_interpreter

    @property
    def grouping_judge(self) -> GeminiGroupingJudge:
        if self._grouping_judge is None:
            self._grouping_judge = GeminiGroupingJudge(
                model=self.semantic_model,
                api_key=_gemini_api_key_or_empty(),
            )
        return self._grouping_judge

    @property
    def routing_judge(self) -> GeminiRoutingJudge:
        if self._routing_judge is None:
            self._routing_judge = GeminiRoutingJudge(
                model=self.semantic_model,
                api_key=_gemini_api_key_or_empty(),
            )
        return self._routing_judge

    def resident_client_names(self) -> list[str]:
        snapshot = self.remote_runtime_snapshot()
        return snapshot.resident_models

    def remote_runtime_snapshot(self) -> RemoteRuntimeSnapshot:
        payload = server_runtime_info(
            self.server_base_url,
            timeout_seconds=min(self.remote_timeout_seconds, 30.0),
        )
        return RemoteRuntimeSnapshot(
            effective_runtime_profile=(
                str(payload.get("effective_runtime_profile"))
                if payload.get("effective_runtime_profile") is not None
                else None
            ),
            runtime_profile_source=(
                str(payload.get("runtime_profile_source"))
                if payload.get("runtime_profile_source") is not None
                else None
            ),
            residency_mode=(
                str(payload.get("residency_mode"))
                if payload.get("residency_mode") is not None
                else None
            ),
            resident_models=[str(item) for item in payload.get("resident_models", [])],
        )


def _gemini_api_key_or_empty() -> str:
    if settings.gemini_api_key is None:
        return ""
    return settings.gemini_api_key.get_secret_value()
