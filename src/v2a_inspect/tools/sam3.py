from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from .remote import post_json
from .types import FrameBatch, Sam3TrackSet


class Sam3ExtractRequest(BaseModel):
    mode: str = "prompt_free"
    frames: list[dict[str, object]] = Field(default_factory=list)
    text_prompt: str | None = None


class Sam3RunpodClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def extract_entities(self, frame_batches: list[FrameBatch]) -> Sam3TrackSet:
        payload = Sam3ExtractRequest(
            mode="prompt_free",
            frames=_build_frame_payload(frame_batches),
        ).model_dump(mode="json")
        response = post_json(
            self.endpoint_url,
            {"input": payload},
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        return _parse_track_set(response, strategy="prompt_free")

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
        response = post_json(
            self.endpoint_url,
            {"input": payload},
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        return _parse_track_set(response, strategy="text_recovery")


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
    response: dict[str, object],
    *,
    strategy: str,
) -> Sam3TrackSet:
    payload = response.get("output", response)
    if not isinstance(payload, dict):
        raise TypeError("SAM3 endpoint returned an invalid payload.")
    normalized_payload = dict(payload)
    normalized_payload.setdefault("provider", "sam3")
    normalized_payload.setdefault("strategy", strategy)
    return Sam3TrackSet.model_validate(normalized_payload)
