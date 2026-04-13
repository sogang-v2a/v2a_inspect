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

from .source_ontology import EMERGENCY_FALLBACK_PROMPTS

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
    def __init__(self, *, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key
        self._llm: BaseChatModel | None = None

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(model=self._model, api_key=self._api_key)
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
        for batch in frame_batches:
            if not batch.frames:
                continue
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
    scored: dict[int, dict[str, list[dict[str, float]]]] = {}
    for batch in frame_batches:
        image_paths = [frame.image_path for frame in batch.frames]
        if not image_paths:
            continue
        extraction_scores = label_client.score_image_labels(
            image_paths=image_paths,
            labels=list(extraction_terms),
        )[:top_k]
        semantic_scores = label_client.score_image_labels(
            image_paths=image_paths,
            labels=list(semantic_terms),
        )[:top_k]
        scored[batch.scene_index] = {
            "extraction_entities": [
                {"label": item.label, "score": round(float(item.score), 4)}
                for item in extraction_scores
            ],
            "semantic_hints": [
                {"label": item.label, "score": round(float(item.score), 4)}
                for item in semantic_scores
            ],
        }
    return scored


def propose_moving_regions(
    frame_batches: Sequence[FrameBatch],
    *,
    output_root: str,
    threshold: float = 0.08,
) -> dict[int, list[RegionProposal]]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    proposals_by_scene: dict[int, list[RegionProposal]] = {}
    for batch in frame_batches:
        proposals: list[RegionProposal] = []
        frames = list(batch.frames)
        for index, (left, right) in enumerate(zip(frames, frames[1:], strict=False)):
            proposal = _proposal_from_frame_pair(
                scene_index=batch.scene_index,
                frame_index=index + 1,
                left_path=left.image_path,
                right_path=right.image_path,
                output_root=root,
                threshold=threshold,
            )
            if proposal is not None:
                proposals.append(proposal)
        proposals_by_scene[batch.scene_index] = proposals
    return proposals_by_scene


def expand_scene_ontology(
    *,
    frame_batches: Sequence[FrameBatch],
    ontology_scores: Mapping[int, Mapping[str, Sequence[Mapping[str, float]]]],
    scene_hypotheses: Mapping[int, SceneHypothesis],
    moving_region_labels: Mapping[int, Sequence[str]],
    top_prompt_count: int = 8,
) -> dict[int, WindowOntologyExpansion]:
    expansions: dict[int, WindowOntologyExpansion] = {}
    for batch in frame_batches:
        scene_index = batch.scene_index
        score_payload = ontology_scores.get(scene_index, {})
        top_entities = [
            str(item["label"])
            for item in list(score_payload.get("extraction_entities", []))[:top_prompt_count]
        ]
        top_hints = [
            str(item["label"])
            for item in list(score_payload.get("semantic_hints", []))[:top_prompt_count]
        ]
        hypothesis = scene_hypotheses.get(scene_index)
        merged_prompts = _dedupe(
            [
                *top_entities,
                *(hypothesis.foreground_entities if hypothesis else []),
                *(hypothesis.candidate_sound_sources if hypothesis else []),
                *list(moving_region_labels.get(scene_index, [])),
            ]
        )[:top_prompt_count]
        semantic_hints = _dedupe(
            [
                *top_hints,
                *(hypothesis.background_environment if hypothesis else []),
                *(hypothesis.interactions if hypothesis else []),
                *(hypothesis.material_cues if hypothesis else []),
            ]
        )[: top_prompt_count * 3]
        expansions[scene_index] = WindowOntologyExpansion(
            extraction_prompts=merged_prompts,
            semantic_hints=semantic_hints,
            provenance={
                "ontology_extraction": top_entities,
                "ontology_semantics": top_hints,
                "gemini_hypothesis": None if hypothesis is None else hypothesis.model_dump(mode="json"),
                "moving_region_labels": list(moving_region_labels.get(scene_index, [])),
            },
        )
    return expansions


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
    verified: dict[int, VerifiedSceneHypothesis] = {}
    crop_label_hints_by_window = crop_label_hints_by_window or {}
    for batch in frame_batches:
        scene_index = batch.scene_index
        expansion = expanded_candidates.get(scene_index, WindowOntologyExpansion())
        score_payload = ontology_scores.get(scene_index, {})
        ontology_scores_by_label = {
            str(item["label"]): float(item["score"])
            for item in score_payload.get("extraction_entities", [])
        }
        support_sources = _support_sources(
            ontology_scores_by_label=ontology_scores_by_label,
            scene_hypothesis=scene_hypotheses.get(scene_index),
            moving_region_labels=moving_region_labels.get(scene_index, []),
            crop_label_hints=crop_label_hints_by_window.get(scene_index, []),
        )
        support_counts = {label: len(sources) for label, sources in support_sources.items()}
        verified_prompts: list[str] = []
        uncertain: list[str] = []
        rejected: list[str] = []
        for label in expansion.extraction_prompts:
            support_count = support_counts.get(label, 0)
            ontology_score = ontology_scores_by_label.get(label, 0.0)
            if support_count >= 2 or ontology_score >= strong_single_source_threshold:
                verified_prompts.append(label)
            elif support_count == 1:
                uncertain.append(label)
            else:
                rejected.append(label)
        verified_prompts = _dedupe(verified_prompts)[:top_prompt_count]
        if not verified_prompts:
            fallback = uncertain[:1] or list(EMERGENCY_FALLBACK_PROMPTS[:1])
            verified_prompts = _dedupe([*fallback, "object"])[:top_prompt_count]
        elif "object" not in verified_prompts:
            verified_prompts.append("object")
        semantic_hints = _dedupe(
            [
                *expansion.semantic_hints,
                *uncertain,
                *list(crop_label_hints_by_window.get(scene_index, [])),
            ]
        )[: top_prompt_count * 3]
        verified[scene_index] = VerifiedSceneHypothesis(
            verified_extraction_prompts=verified_prompts,
            verified_semantic_hints=semantic_hints,
            rejected_hypotheses=_dedupe(rejected),
            uncertain_hypotheses=_dedupe(uncertain),
            support_counts={key: int(value) for key, value in support_counts.items()},
            verification_rationale=_verification_rationale(
                verified_prompts=verified_prompts,
                uncertain_hypotheses=uncertain,
                rejected_hypotheses=rejected,
                support_sources=support_sources,
            ),
        )
    return verified


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
