from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageChops, ImageOps, ImageStat
from pydantic import BaseModel, Field

from v2a_inspect.runtime import build_llm
from v2a_inspect.tools.types import FrameBatch


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class SceneHypothesis(BaseModel):
    foreground_entities: list[str] = Field(default_factory=list)
    background_environment: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    material_cues: list[str] = Field(default_factory=list)
    candidate_sound_sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_frame_indices: list[int] = Field(default_factory=list)


class RegionProposal(BaseModel):
    scene_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    motion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(default_factory=list, min_length=4, max_length=4)
    crop_path: str | None = None
    label_hint: str | None = None


class WindowOntologyExpansion(BaseModel):
    extraction_prompts: list[str] = Field(default_factory=list)
    semantic_hints: list[str] = Field(default_factory=list)
    provenance: dict[str, object] = Field(default_factory=dict)


class VerifiedSceneHypothesis(BaseModel):
    verified_extraction_prompts: list[str] = Field(default_factory=list)
    verified_semantic_hints: list[str] = Field(default_factory=list)
    rejected_hypotheses: list[str] = Field(default_factory=list)
    uncertain_hypotheses: list[str] = Field(default_factory=list)
    support_counts: dict[str, int] = Field(default_factory=dict)
    verification_rationale: str = ""


class _SceneHypothesisPayload(BaseModel):
    foreground_entities: list[str] = Field(default_factory=list)
    background_environment: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    material_cues: list[str] = Field(default_factory=list)
    candidate_sound_sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_frame_indices: list[int] = Field(default_factory=list)


class GeminiSceneHypothesisProposer:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        max_retries: int = 1,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._llm: BaseChatModel | None = None
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

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
        ontology_terms: Sequence[str],
    ) -> dict[int, SceneHypothesis]:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured_llm = self.llm.with_structured_output(
            _SceneHypothesisPayload,
            method="json_schema",
        )
        results: dict[int, SceneHypothesis] = {}
        ontology_preview = list(ontology_terms)[:96]
        proposal_failed = False
        for batch in frame_batches:
            if not batch.frames:
                continue
            if proposal_failed:
                break
            content: list[object] = [
                {
                    "type": "text",
                    "text": (
                        "Propose visible sound-source hypotheses from these sampled frames only. "
                        "Do not infer from audio. Return visible noun-like entities worth extracting, "
                        "environment cues, material cues, interactions, and candidate sound sources.\n\n"
                        f"Allowed ontology hints: {json.dumps(ontology_preview, ensure_ascii=False)}"
                    ),
                }
            ]
            for index, frame in enumerate(batch.frames[:3]):
                content.append(
                    {
                        "type": "text",
                        "text": f"Frame {index} at {frame.timestamp_seconds:.2f}s",
                    }
                )
                content.append(_image_block(frame.image_path))
            prompt = [
                SystemMessage(
                    content=(
                        "You infer visible source hypotheses for a silent-video Foley analysis pipeline. "
                        "Use only the provided images. Prefer concrete visible entities and interactions."
                    )
                ),
                HumanMessage(content=content),
            ]
            try:
                payload = structured_llm.invoke(prompt)
            except Exception:  # noqa: BLE001
                proposal_failed = True
                continue
            if not isinstance(payload, _SceneHypothesisPayload):
                payload = _SceneHypothesisPayload.model_validate(payload)
            results[batch.scene_index] = SceneHypothesis.model_validate(payload.model_dump())
        return results


def score_scene_ontology(
    frame_batches: Sequence[FrameBatch],
    label_client: object,
    *,
    extraction_terms: Sequence[str],
    semantic_terms: Sequence[str],
    top_k: int = 8,
) -> dict[int, dict[str, list[dict[str, float]]]]:
    del frame_batches, label_client, extraction_terms, semantic_terms, top_k
    return {}


def propose_moving_regions(
    frame_batches: Sequence[FrameBatch],
    *,
    output_root: str,
    threshold: float = 0.08,
) -> dict[int, list[RegionProposal]]:
    del frame_batches, output_root, threshold
    return {}


def expand_scene_ontology(
    *,
    frame_batches: Sequence[FrameBatch],
    ontology_scores: Mapping[int, Mapping[str, Sequence[Mapping[str, float]]]],
    scene_hypotheses: Mapping[int, SceneHypothesis],
    moving_region_labels: Mapping[int, Sequence[str]],
    top_prompt_count: int = 8,
) -> dict[int, WindowOntologyExpansion]:
    del ontology_scores, scene_hypotheses, moving_region_labels, top_prompt_count
    return {
        batch.scene_index: WindowOntologyExpansion(
            extraction_prompts=[],
            semantic_hints=[],
            provenance={},
        )
        for batch in frame_batches
    }


def verify_scene_hypotheses(
    *,
    frame_batches: Sequence[FrameBatch],
    ontology_scores: Mapping[int, Mapping[str, Sequence[Mapping[str, float]]]],
    scene_hypotheses: Mapping[int, SceneHypothesis],
    moving_region_labels: Mapping[int, Sequence[str]],
    expanded_candidates: Mapping[int, WindowOntologyExpansion],
    crop_label_hints_by_window: Mapping[int, Sequence[str]] | None = None,
    top_prompt_count: int = 8,
    strong_single_source_threshold: float = 0.08,
) -> dict[int, VerifiedSceneHypothesis]:
    del (
        ontology_scores,
        scene_hypotheses,
        moving_region_labels,
        expanded_candidates,
        crop_label_hints_by_window,
        top_prompt_count,
        strong_single_source_threshold,
    )
    return {
        batch.scene_index: VerifiedSceneHypothesis(
            verified_extraction_prompts=[],
            verified_semantic_hints=[],
            rejected_hypotheses=[],
            uncertain_hypotheses=[],
            support_counts={},
            verification_rationale="",
        )
        for batch in frame_batches
    }


def label_moving_region_crops(
    *,
    proposals_by_scene: Mapping[int, Sequence[RegionProposal]],
    label_client: object,
    label_vocabulary: Sequence[str],
    top_k: int = 2,
) -> dict[int, list[str]]:
    labels_by_scene: dict[int, list[str]] = {}
    for scene_index, proposals in proposals_by_scene.items():
        scene_labels: list[str] = []
        for proposal in proposals:
            if proposal.crop_path is None:
                continue
            scored = label_client.score_image_labels(
                image_paths=[proposal.crop_path],
                labels=list(label_vocabulary),
            )[:top_k]
            for item in scored:
                scene_labels.append(item.label)
        labels_by_scene[scene_index] = _dedupe(scene_labels)
    return labels_by_scene


def _support_sources(
    *,
    ontology_scores_by_label: Mapping[str, float],
    scene_hypothesis: SceneHypothesis | None,
    moving_region_labels: Sequence[str],
    crop_label_hints: Sequence[str],
) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {}
    for label, score in ontology_scores_by_label.items():
        if score >= 0.08:
            sources.setdefault(label, set()).add("ontology")
    if scene_hypothesis is not None:
        gemini_terms = [
            *scene_hypothesis.foreground_entities,
            *scene_hypothesis.candidate_sound_sources,
        ]
        for label in gemini_terms:
            sources.setdefault(label, set()).add("gemini")
    for label in moving_region_labels:
        sources.setdefault(label, set()).add("motion")
    for label in crop_label_hints:
        sources.setdefault(label, set()).add("crop")
    return sources


def _verification_rationale(
    *,
    verified_prompts: Sequence[str],
    uncertain_hypotheses: Sequence[str],
    rejected_hypotheses: Sequence[str],
    support_sources: Mapping[str, set[str]],
) -> str:
    parts: list[str] = []
    if verified_prompts:
        parts.append(
            "verified=" + ", ".join(
                f"{label}<{'+'.join(sorted(support_sources.get(label, set())))}>"
                for label in verified_prompts
            )
        )
    if uncertain_hypotheses:
        parts.append("uncertain=" + ", ".join(sorted(set(uncertain_hypotheses))))
    if rejected_hypotheses:
        parts.append("rejected=" + ", ".join(sorted(set(rejected_hypotheses))))
    return " | ".join(parts) if parts else "no strong hypothesis evidence"


def _proposal_from_frame_pair(
    *,
    scene_index: int,
    frame_index: int,
    left_path: str,
    right_path: str,
    output_root: Path,
    threshold: float,
) -> RegionProposal | None:
    with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
        left_gray = ImageOps.grayscale(left_image.convert("RGB")).resize((128, 128))
        right_gray = ImageOps.grayscale(right_image.convert("RGB")).resize((128, 128))
        diff = ImageChops.difference(left_gray, right_gray)
        score = float(ImageStat.Stat(diff).mean[0] / 255.0)
        if score < threshold:
            return None
        bbox = diff.getbbox()
        if bbox is None:
            return None
        scale_x = max(left_image.width / 128.0, 1.0)
        scale_y = max(left_image.height / 128.0, 1.0)
        bbox_xyxy = [
            round(bbox[0] * scale_x, 3),
            round(bbox[1] * scale_y, 3),
            round(bbox[2] * scale_x, 3),
            round(bbox[3] * scale_y, 3),
        ]
        crop_path = output_root / f"scene_{scene_index:04d}_motion_{frame_index:02d}.jpg"
        cropped = right_image.crop((
            max(int(bbox_xyxy[0]), 0),
            max(int(bbox_xyxy[1]), 0),
            max(int(bbox_xyxy[2]), 1),
            max(int(bbox_xyxy[3]), 1),
        ))
        cropped.save(crop_path, format="JPEG")
    return RegionProposal(
        scene_index=scene_index,
        frame_index=frame_index,
        motion_score=round(min(max(score, 0.0), 1.0), 4),
        bbox_xyxy=bbox_xyxy,
        crop_path=str(crop_path),
    )


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _image_block(image_path: str) -> Mapping[str, object]:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }
