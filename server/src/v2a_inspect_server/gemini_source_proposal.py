from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.runtime import build_llm
from v2a_inspect.tools.types import FrameBatch

from .scene_hypotheses import RegionProposal, _image_block

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class WindowSourceProposal(BaseModel):
    visible_sources: list[str] = Field(default_factory=list)
    background_sources: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    materials_surfaces: list[str] = Field(default_factory=list)
    uncertain_regions: list[str] = Field(default_factory=list)
    salient_regions: list[str] = Field(default_factory=list)
    supporting_frame_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class _WindowSourceProposalPayload(WindowSourceProposal):
    pass


class GeminiSourceProposer:
    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str,
        max_retries: int = 1,
        timeout_seconds: float = 45.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._llm: BaseChatModel | None = None
        self.last_error_message: str | None = None

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(
                model=self._model,
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout_seconds=self._timeout_seconds,
            )
        return self._llm

    def propose(
        self,
        *,
        frame_batches: Sequence[FrameBatch],
        storyboard_path: str | None,
        moving_regions_by_scene: Mapping[int, Sequence[RegionProposal]] | None = None,
    ) -> dict[int, WindowSourceProposal]:
        from langchain_core.messages import HumanMessage, SystemMessage

        self.last_error_message = None
        moving_regions_by_scene = moving_regions_by_scene or {}
        structured_llm = self.llm.with_structured_output(
            _WindowSourceProposalPayload,
            method="json_schema",
        )
        proposals: dict[int, WindowSourceProposal] = {}
        storyboard_block = _image_block(storyboard_path) if storyboard_path else None
        for batch in frame_batches:
            if not batch.frames:
                continue
            content: list[object] = [
                {
                    "type": "text",
                    "text": (
                        "Infer visible potential sound-producing sources from this silent-video window. "
                        "This is open-world: do not restrict yourself to a fixed vocabulary. "
                        "Use only visible evidence from the provided sampled frames, storyboard, and motion-region crops. "
                        "Return concrete noun phrases for visible sources, background/environment sources, "
                        "interactions, material/surface cues, uncertain but plausible regions, and salient regions worth extracting."
                    ),
                }
            ]
            if storyboard_block is not None:
                content.append({"type": "text", "text": "Global storyboard context"})
                content.append(storyboard_block)
            for index, frame in enumerate(batch.frames[:4]):
                content.append(
                    {
                        "type": "text",
                        "text": f"Window frame {index} at {frame.timestamp_seconds:.2f}s",
                    }
                )
                content.append(_image_block(frame.image_path))
            for index, proposal in enumerate(list(moving_regions_by_scene.get(batch.scene_index, []))[:3]):
                if proposal.crop_path:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                f"Motion-region crop {index} from frame {proposal.frame_index} "
                                f"motion_score={proposal.motion_score:.3f} bbox={json.dumps(proposal.bbox_xyxy)}"
                            ),
                        }
                    )
                    content.append(_image_block(proposal.crop_path))
            prompt = [
                SystemMessage(
                    content=(
                        "You are performing semantic source proposal for a video-only Foley inspection pipeline. "
                        "Do not infer from audio. Prefer visible concrete entities, interacting objects, and visible ambience context."
                    )
                ),
                HumanMessage(content=content),
            ]
            try:
                payload = structured_llm.invoke(prompt)
            except Exception as exc:  # noqa: BLE001
                if self.last_error_message is None:
                    self.last_error_message = (
                        f"Gemini source proposal failed: {type(exc).__name__}: {str(exc)[:240]}"
                    )
                continue
            if not isinstance(payload, _WindowSourceProposalPayload):
                payload = _WindowSourceProposalPayload.model_validate(payload)
            proposals[batch.scene_index] = WindowSourceProposal.model_validate(payload.model_dump())
        return proposals
