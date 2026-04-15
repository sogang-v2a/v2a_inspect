from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.runtime import build_llm
from v2a_inspect.tools.types import Sam3RegionSeed
from v2a_inspect.tools.types import FrameBatch

from .gemini_source_proposal import SourceCard, WindowSourceProposal
from .scene_hypotheses import RegionProposal, _image_block

if TYPE_CHECKING:
    from .embeddings import LabelClient
    from langchain_core.language_models import BaseChatModel


class PhraseGroundingEvidence(BaseModel):
    phrase: str
    frame_score_max: float = Field(default=0.0, ge=0.0, le=1.0)
    frame_score_mean: float = Field(default=0.0, ge=0.0, le=1.0)
    motion_crop_score_max: float = Field(default=0.0, ge=0.0, le=1.0)


class GroundedSourceCard(BaseModel):
    source_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    source_kind_candidate: str = "unknown"
    sound_relevance: str = "unknown"
    interaction: str | None = None
    material_or_surface: str | None = None
    region_refs: list[int] = Field(default_factory=list)
    supporting_frame_indices: list[int] = Field(default_factory=list)
    extraction_prompt: str | None = None
    semantic_hints: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class GroundedWindowProposal(BaseModel):
    grounded_source_cards: list[GroundedSourceCard] = Field(default_factory=list)
    region_seeds: list[Sam3RegionSeed] = Field(default_factory=list)
    extraction_prompts: list[str] = Field(default_factory=list)
    semantic_hints: list[str] = Field(default_factory=list)
    rejected_phrases: list[str] = Field(default_factory=list)
    unresolved_phrases: list[str] = Field(default_factory=list)
    phrase_evidence: list[PhraseGroundingEvidence] = Field(default_factory=list)
    rationale: str = ""


class _GroundedWindowProposalPayload(BaseModel):
    grounded_source_cards: list[GroundedSourceCard] = Field(default_factory=list)
    extraction_prompts: list[str] = Field(default_factory=list)
    semantic_hints: list[str] = Field(default_factory=list)
    rejected_phrases: list[str] = Field(default_factory=list)
    unresolved_phrases: list[str] = Field(default_factory=list)
    rationale: str = ""


class GeminiProposalGrounder:
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

    def ground(
        self,
        *,
        frame_batches: Sequence[FrameBatch],
        storyboard_path: str | None,
        proposals_by_scene: Mapping[int, WindowSourceProposal],
        moving_regions_by_scene: Mapping[int, Sequence[RegionProposal]] | None,
        label_client: "LabelClient",
    ) -> dict[int, GroundedWindowProposal]:
        from langchain_core.messages import HumanMessage, SystemMessage

        self.last_error_message = None
        moving_regions_by_scene = moving_regions_by_scene or {}
        structured_llm = self.llm.with_structured_output(
            _GroundedWindowProposalPayload,
            method="json_schema",
        )
        grounded: dict[int, GroundedWindowProposal] = {}
        storyboard_block = _image_block(storyboard_path) if storyboard_path else None
        for batch in frame_batches:
            proposal = proposals_by_scene.get(batch.scene_index)
            if proposal is None:
                continue
            proposal_regions = list(moving_regions_by_scene.get(batch.scene_index, []))
            candidate_phrases = _dedupe(
                [
                    *[
                        label
                        for card in proposal.source_cards
                        for label in [card.source_name, *card.aliases]
                    ],
                    *proposal.visible_sources,
                    *proposal.background_sources,
                    *proposal.interactions,
                    *proposal.materials_surfaces,
                    *proposal.uncertain_regions,
                ]
            )
            evidence = _score_candidate_phrases(
                candidate_phrases,
                frame_image_paths=[frame.image_path for frame in batch.frames],
                motion_crop_paths=[
                    region.crop_path
                    for region in proposal_regions
                    if region.crop_path
                ],
                label_client=label_client,
            )
            if not candidate_phrases:
                grounded[batch.scene_index] = GroundedWindowProposal(
                    grounded_source_cards=[],
                    region_seeds=[],
                    extraction_prompts=[],
                    semantic_hints=[],
                    rejected_phrases=[],
                    unresolved_phrases=[],
                    phrase_evidence=evidence,
                    rationale="no open-world proposal phrases available",
                )
                continue
            content: list[object] = [
                {
                    "type": "text",
                    "text": (
                        "Ground source-proposal phrases for silent-video extraction. "
                        "Use only the provided frames, storyboard, motion-region crops, and score summaries. "
                        "Choose extraction prompts only for phrases that are visually grounded enough to seed SAM extraction. "
                        "Promote region-grounded source cards into extraction seeds whenever the card points to a plausible visible region. "
                        "Keep environment/material/interaction terms as semantic hints unless they denote a concrete visible source."
                    ),
                },
                {
                    "type": "text",
                    "text": (
                        "Source cards from the proposal stage:\n"
                        + json.dumps(
                            [card.model_dump(mode='json') for card in proposal.source_cards],
                            indent=2,
                            ensure_ascii=False,
                        )
                    ),
                },
                {
                    "type": "text",
                    "text": (
                        "Structured candidate evidence:\n"
                        + json.dumps([item.model_dump(mode='json') for item in evidence], indent=2, ensure_ascii=False)
                    ),
                },
            ]
            if storyboard_block is not None:
                content.append({"type": "text", "text": "Global storyboard context"})
                content.append(storyboard_block)
            for index, frame in enumerate(batch.frames[:4]):
                content.append({"type": "text", "text": f"Window frame {index}"})
                content.append(_image_block(frame.image_path))
            for index, proposal_region in enumerate(list(moving_regions_by_scene.get(batch.scene_index, []))[:3]):
                if proposal_region.crop_path:
                    content.append({"type": "text", "text": f"Motion-region crop {index}"})
                    content.append(_image_block(proposal_region.crop_path))
            prompt = [
                SystemMessage(
                    content=(
                        "You ground open-world visual source proposals for a video-only Foley pipeline. "
                        "Do not add vocabulary not supported by the visible evidence unless you place it in unresolved_phrases."
                    )
                ),
                HumanMessage(content=content),
            ]
            try:
                payload = structured_llm.invoke(prompt)
            except Exception as exc:  # noqa: BLE001
                if self.last_error_message is None:
                    self.last_error_message = (
                        f"Gemini proposal grounding failed: {type(exc).__name__}: {str(exc)[:240]}"
                    )
                fallback_cards = _fallback_grounded_source_cards(
                    proposal.source_cards,
                    evidence=evidence,
                )
                grounded[batch.scene_index] = GroundedWindowProposal(
                    grounded_source_cards=fallback_cards,
                    region_seeds=_region_seeds_from_cards(
                        batch.scene_index,
                        fallback_cards,
                        proposal_regions,
                    ),
                    extraction_prompts=_prompts_from_cards(fallback_cards),
                    semantic_hints=[],
                    rejected_phrases=[],
                    unresolved_phrases=candidate_phrases,
                    phrase_evidence=evidence,
                    rationale="grounding unavailable",
                )
                continue
            if not isinstance(payload, _GroundedWindowProposalPayload):
                payload = _GroundedWindowProposalPayload.model_validate(payload)
            grounded_cards = list(payload.grounded_source_cards)
            if not grounded_cards:
                grounded_cards = _fallback_grounded_source_cards(
                    proposal.source_cards,
                    evidence=evidence,
                )
            extraction_prompts = _dedupe(
                [*payload.extraction_prompts, *_prompts_from_cards(grounded_cards)]
            )
            semantic_hints = _dedupe(
                [
                    *payload.semantic_hints,
                    *[
                        hint
                        for card in grounded_cards
                        for hint in card.semantic_hints
                    ],
                ]
            )
            grounded[batch.scene_index] = GroundedWindowProposal(
                grounded_source_cards=grounded_cards,
                region_seeds=_region_seeds_from_cards(
                    batch.scene_index,
                    grounded_cards,
                    proposal_regions,
                ),
                extraction_prompts=extraction_prompts,
                semantic_hints=semantic_hints,
                rejected_phrases=_dedupe(payload.rejected_phrases),
                unresolved_phrases=_dedupe(payload.unresolved_phrases),
                phrase_evidence=evidence,
                rationale=payload.rationale,
            )
        return grounded


def _score_candidate_phrases(
    candidate_phrases: Sequence[str],
    *,
    frame_image_paths: Sequence[str],
    motion_crop_paths: Sequence[str],
    label_client: "LabelClient",
) -> list[PhraseGroundingEvidence]:
    phrases = _dedupe(candidate_phrases)
    if not phrases:
        return []
    frame_scores = label_client.score_image_labels(
        image_paths=list(frame_image_paths),
        labels=list(phrases),
    )
    motion_scores = (
        label_client.score_image_labels(
            image_paths=list(motion_crop_paths),
            labels=list(phrases),
        )
        if motion_crop_paths
        else []
    )
    frame_score_by_phrase = {item.label: float(item.score) for item in frame_scores}
    motion_score_by_phrase = {item.label: float(item.score) for item in motion_scores}
    evidence: list[PhraseGroundingEvidence] = []
    for phrase in phrases:
        frame_score = frame_score_by_phrase.get(phrase, 0.0)
        motion_score = motion_score_by_phrase.get(phrase, 0.0)
        evidence.append(
            PhraseGroundingEvidence(
                phrase=phrase,
                frame_score_max=round(frame_score, 4),
                frame_score_mean=round(mean([frame_score]), 4),
                motion_crop_score_max=round(motion_score, 4),
            )
        )
    evidence.sort(
        key=lambda item: (item.frame_score_max, item.motion_crop_score_max),
        reverse=True,
    )
    return evidence


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _prompts_from_cards(cards: Sequence[GroundedSourceCard]) -> list[str]:
    prompts: list[str] = []
    for card in cards:
        prompt = (card.extraction_prompt or card.source_name).strip().lower()
        if prompt:
            prompts.append(prompt)
    return _dedupe(prompts)


def _fallback_grounded_source_cards(
    source_cards: Sequence[SourceCard],
    *,
    evidence: Sequence[PhraseGroundingEvidence],
) -> list[GroundedSourceCard]:
    if not source_cards:
        return []
    best_score_by_phrase = {
        item.phrase: max(item.frame_score_max, item.motion_crop_score_max)
        for item in evidence
    }
    promoted: list[GroundedSourceCard] = []
    for card in source_cards:
        labels = [card.source_name, *card.aliases]
        best_score = max(
            (best_score_by_phrase.get(label.strip().lower(), 0.0) for label in labels if label.strip()),
            default=0.0,
        )
        if not card.region_refs and best_score < 0.22:
            continue
        promoted.append(
            GroundedSourceCard(
                source_name=card.source_name,
                aliases=list(card.aliases),
                source_kind_candidate=card.source_kind_candidate,
                sound_relevance=card.sound_relevance,
                interaction=card.interaction,
                material_or_surface=card.material_or_surface,
                region_refs=list(card.region_refs),
                supporting_frame_indices=list(card.supporting_frame_indices),
                extraction_prompt=card.source_name,
                semantic_hints=_dedupe(
                    [card.material_or_surface or "", card.interaction or ""]
                ),
                confidence=round(best_score if best_score > 0.0 else card.confidence, 4),
                rationale="fallback promotion from region-backed proposal evidence",
            )
        )
    promoted.sort(key=lambda item: item.confidence, reverse=True)
    return promoted[:3]


def _region_seeds_from_cards(
    scene_index: int,
    cards: Sequence[GroundedSourceCard],
    regions: Sequence[RegionProposal],
) -> list[Sam3RegionSeed]:
    seeds: list[Sam3RegionSeed] = []
    for card in cards:
        for region_index in card.region_refs:
            if region_index < 0 or region_index >= len(regions):
                continue
            region = regions[region_index]
            seeds.append(
                Sam3RegionSeed(
                    scene_index=scene_index,
                    region_index=region_index,
                    bbox_xyxy=list(region.bbox_xyxy),
                    label_hint=(card.extraction_prompt or card.source_name).strip().lower() or None,
                    crop_path=region.crop_path,
                    confidence=round(max(card.confidence, region.motion_score), 4),
                )
            )
    deduped: list[Sam3RegionSeed] = []
    seen: set[tuple[int, tuple[float, float, float, float], str | None]] = set()
    for seed in seeds:
        key = (seed.region_index, tuple(seed.bbox_xyxy), seed.label_hint)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seed)
    return deduped
