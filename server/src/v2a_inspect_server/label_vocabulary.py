from __future__ import annotations

from collections.abc import Mapping


def dynamic_label_vocabulary(
    verified_hypotheses_by_window: Mapping[int, dict[str, object]],
    scene_hypotheses_by_window: Mapping[int, dict[str, object]],
) -> list[str]:
    labels: list[str] = []
    for payload in verified_hypotheses_by_window.values():
        labels.extend(_string_list(payload.get("extraction_prompts")))
        labels.extend(_string_list(payload.get("semantic_hints")))
        for card in _dict_list(payload.get("grounded_source_cards")):
            labels.extend(
                _string_list(
                    [
                        card.get("source_name"),
                        *(card.get("aliases") or []),
                        card.get("extraction_prompt"),
                    ]
                )
            )
    for payload in scene_hypotheses_by_window.values():
        labels.extend(_string_list(payload.get("visible_sources")))
        labels.extend(_string_list(payload.get("background_sources")))
        labels.extend(_string_list(payload.get("uncertain_regions")))
        for card in _dict_list(payload.get("source_cards")):
            labels.extend(
                _string_list(
                    [card.get("source_name"), *(card.get("aliases") or [])]
                )
            )
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = label.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
