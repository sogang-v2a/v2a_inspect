from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from v2a_inspect.tools import RemoteGpuPolicy
from v2a_inspect.tools.types import FrameBatch, Sam3TrackSet

from .execution import execute_service
from .providers import GpuProvider, ProviderMode, ProviderServiceConfig


class Sam3ExtractRequest(BaseModel):
    mode: str = "prompt_free"
    frames: list[dict[str, object]] = Field(default_factory=list)
    text_prompt: str | None = None


class Sam3Client:
    def __init__(
        self,
        *,
        provider: GpuProvider,
        service: str,
        gpu_policy: RemoteGpuPolicy,
        mode: ProviderMode = "sync_endpoint",
    ) -> None:
        self.provider = provider
        self.service = ProviderServiceConfig(name=service, route=service, mode=mode)
        self.gpu_policy = gpu_policy

    def extract_entities(self, frame_batches: list[FrameBatch]) -> Sam3TrackSet:
        payload = Sam3ExtractRequest(
            mode="prompt_free",
            frames=_build_frame_payload(frame_batches),
        ).model_dump(mode="json")
        result = execute_service(
            self.provider,
            self.service,
            payload,
            gpu_policy=self.gpu_policy,
        )
        return _parse_track_set(result.payload, strategy="prompt_free")

    def recover_with_text_prompt(
        self,
        frame_batches: list[FrameBatch],
        *,
        text_prompt: str,
    ) -> Sam3TrackSet:
        payload = Sam3ExtractRequest(
            mode="text_recovery",
            text_prompt=text_prompt,
            frames=_build_frame_payload(frame_batches),
        ).model_dump(mode="json")
        result = execute_service(
            self.provider,
            self.service,
            payload,
            gpu_policy=self.gpu_policy,
        )
        return _parse_track_set(result.payload, strategy="text_recovery")


Sam3RunpodClient = Sam3Client


def _build_frame_payload(frame_batches: list[FrameBatch]) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for batch in frame_batches:
        for frame in batch.frames:
            frames.append(
                {
                    "scene_index": frame.scene_index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "image_base64": base64.b64encode(
                        Path(frame.image_path).read_bytes()
                    ).decode("ascii"),
                }
            )
    return frames


def _parse_track_set(
    payload: dict[str, object],
    *,
    strategy: str,
) -> Sam3TrackSet:
    normalized_payload = dict(payload)
    normalized_payload.setdefault("provider", "sam3")
    normalized_payload.setdefault("strategy", strategy)
    return Sam3TrackSet.model_validate(normalized_payload)
