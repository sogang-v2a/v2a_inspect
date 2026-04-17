from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
from v2a_inspect.contracts import AmbienceBed, LabelCandidate, PhysicalSourceTrack, SoundEventSegment, TrackCrop
from v2a_inspect.runtime import build_llm
from v2a_inspect.tools.types import Sam3EntityTrack

from .scene_hypotheses import _image_block

if TYPE_CHECKING:
    from v2a_inspect.runtime import StructuredChatModel


class SourceSemanticInterpretation(BaseModel):
    canonical_label: str | None = None
    source_kind: Literal[
        "foreground",
        "background_region",
        "ambience_region",
        "unknown",
    ] = "unknown"
    audibility_state: Literal[
        "audible_active",
        "visible_but_silent",
        "background_region",
        "ambience_region",
        "unknown",
    ] = "unknown"
    event_label: str | None = None
    material_or_surface: str | None = None
    intensity: str | None = None
    texture: str | None = None
    pattern: str | None = None
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _SourceSemanticPayload(SourceSemanticInterpretation):
    pass


class GeminiSourceSemanticsInterpreter:
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
        self._llm: StructuredChatModel | None = None

    @property
    def llm(self) -> StructuredChatModel:
        if self._llm is None:
            self._llm = build_llm(
                model=self._model,
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout_seconds=self._timeout_seconds,
            )
        return self._llm

    def interpret_source(
        self,
        *,
        source: PhysicalSourceTrack,
        representative_crop_paths: Sequence[str],
        supporting_frame_paths: Sequence[str],
        track_summaries: Sequence[Mapping[str, object]],
    ) -> SourceSemanticInterpretation | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        content: list[str | dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "Interpret one provisional physical source for a silent-video Foley pipeline. "
                    "Decide whether it is foreground, ambience_region, background_region, or unknown, "
                    "and separately decide whether it is audible_active, visible_but_silent, background_region, ambience_region, or unknown. "
                    "Return a canonical visible source label when possible, and a concise event label only if the visual evidence supports an active sound-making event."
                ),
            },
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "source_id": source.source_id,
                        "spans": source.spans,
                        "window_refs": source.window_refs,
                        "identity_confidence": source.identity_confidence,
                        "track_summaries": list(track_summaries),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            },
        ]
        for index, image_path in enumerate(list(representative_crop_paths)[:4]):
            content.append({"type": "text", "text": f"Representative crop {index}"})
            content.append(_image_block(image_path))
        for index, image_path in enumerate(list(supporting_frame_paths)[:3]):
            content.append({"type": "text", "text": f"Supporting frame {index}"})
            content.append(_image_block(image_path))
        prompt = [
            SystemMessage(
                content=(
                    "You perform semantic interpretation from visual evidence only. "
                    "Do not infer from audio. Leave fields empty when the evidence is insufficient."
                )
            ),
            HumanMessage(content=content),
        ]
        structured_llm = self.llm.with_structured_output(
            _SourceSemanticPayload,
            method="json_schema",
        )
        try:
            payload = structured_llm.invoke(prompt)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, _SourceSemanticPayload):
            payload = _SourceSemanticPayload.model_validate(payload)
        return SourceSemanticInterpretation.model_validate(payload.model_dump())


def build_source_and_event_semantics(
    *,
    physical_sources: Sequence[PhysicalSourceTrack],
    tracks_by_id: Mapping[str, Sam3EntityTrack],
    track_crops: Sequence[TrackCrop],
    evidence_windows: Sequence[object],
    interpreter: GeminiSourceSemanticsInterpreter | None,
) -> dict[str, object]:
    crops_by_track: dict[str, list[str]] = defaultdict(list)
    for crop in track_crops:
        crops_by_track[crop.track_id].append(crop.crop_path)

    frames_by_window: dict[str, list[str]] = {
        getattr(window, "window_id", ""): list(getattr(window, "sampled_frame_ids", []))
        for window in evidence_windows
    }
    updated_sources: list[PhysicalSourceTrack] = []
    sound_events: list[SoundEventSegment] = []
    ambience_beds: list[AmbienceBed] = []

    for source in physical_sources:
        representative_crop_paths = _representative_crop_paths(source, crops_by_track)
        supporting_frame_paths = _supporting_frame_paths(source, frames_by_window)
        track_summaries = [
            _track_summary(tracks_by_id[track_id])
            for track_id in source.track_refs
            if track_id in tracks_by_id
        ]
        interpretation = (
            interpreter.interpret_source(
                source=source,
                representative_crop_paths=representative_crop_paths,
                supporting_frame_paths=supporting_frame_paths,
                track_summaries=track_summaries,
            )
            if interpreter is not None
            else None
        )
        updated_source = source.model_copy(deep=True)
        if interpretation is not None and interpretation.canonical_label:
            label_candidate = LabelCandidate(
                label=interpretation.canonical_label,
                score=round(max(interpretation.confidence, 0.01), 4),
            )
            existing = [candidate for candidate in updated_source.label_candidates if candidate.label != label_candidate.label]
            updated_source.label_candidates = [label_candidate, *existing]
        if interpretation is not None and interpretation.source_kind in {
            "foreground",
            "background_region",
            "ambience_region",
            "unknown",
        }:
            updated_source.kind = interpretation.source_kind
        if interpretation is not None and interpretation.audibility_state in {
            "audible_active",
            "visible_but_silent",
            "background_region",
            "ambience_region",
            "unknown",
        }:
            updated_source.audibility_state = interpretation.audibility_state
        updated_sources.append(updated_source)

        if updated_source.kind in {"ambience_region", "background_region"}:
            ambience_beds.append(
                AmbienceBed(
                    ambience_id=f"ambience-{updated_source.source_id}",
                    start_time=min((span[0] for span in updated_source.spans), default=0.0),
                    end_time=max((span[1] for span in updated_source.spans), default=0.0),
                    environment_type=(interpretation.canonical_label if interpretation and interpretation.canonical_label else ""),
                    acoustic_profile=(interpretation.event_label if interpretation and interpretation.event_label else ""),
                    confidence=round(interpretation.confidence if interpretation else updated_source.identity_confidence, 4),
                )
            )
            continue

        if updated_source.audibility_state != "audible_active":
            continue

        for span_index, (span_start, span_end) in enumerate(updated_source.spans):
            sound_events.append(
                SoundEventSegment(
                    event_id=f"event-{updated_source.source_id}-{span_index:02d}",
                    source_id=updated_source.source_id,
                    start_time=span_start,
                    end_time=span_end,
                    event_type=(interpretation.event_label if interpretation and interpretation.event_label else ""),
                    material_or_surface=(interpretation.material_or_surface if interpretation else None),
                    intensity=(interpretation.intensity if interpretation else None),
                    texture=(interpretation.texture if interpretation else None),
                    pattern=(interpretation.pattern if interpretation else None),
                    confidence=round(interpretation.confidence if interpretation else updated_source.identity_confidence, 4),
                )
            )

    return {
        "physical_sources": updated_sources,
        "sound_events": sound_events,
        "ambience_beds": ambience_beds,
    }


def _representative_crop_paths(
    source: PhysicalSourceTrack,
    crops_by_track: Mapping[str, Sequence[str]],
) -> list[str]:
    images: list[str] = []
    for track_id in source.track_refs:
        for crop_path in crops_by_track.get(track_id, [])[:2]:
            if crop_path not in images:
                images.append(crop_path)
    return images[:4]


def _supporting_frame_paths(
    source: PhysicalSourceTrack,
    frames_by_window: Mapping[str, Sequence[str]],
) -> list[str]:
    frames: list[str] = []
    for window_id in source.window_refs:
        for frame_path in list(frames_by_window.get(window_id, []))[:1]:
            if frame_path not in frames:
                frames.append(frame_path)
    return frames[:3]


def _track_summary(track: Sam3EntityTrack) -> dict[str, object]:
    return {
        "track_id": track.track_id,
        "scene_index": track.scene_index,
        "start_seconds": track.start_seconds,
        "end_seconds": track.end_seconds,
        "confidence": track.confidence,
        "label_hint": track.label_hint,
        "motion_score": track.features.motion_score,
        "interaction_score": track.features.interaction_score,
        "continuity_score": track.features.continuity_score,
    }
