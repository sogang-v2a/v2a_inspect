from __future__ import annotations

from v2a_inspect.contracts import MultitrackDescriptionBundle, ValidationIssue


def validate_bundle(bundle: MultitrackDescriptionBundle) -> list[ValidationIssue]:
    del bundle
    return []


def _is_suspicious_generation_merge(
    *,
    source_ids: set[str],
    sources_by_id: dict[str, object],
) -> bool:
    sources = [sources_by_id[source_id] for source_id in sorted(source_ids) if source_id in sources_by_id]
    if len(sources) < 2:
        return False
    labels = {
        source.label_candidates[0].label
        for source in sources
        if getattr(source, "label_candidates", None)
    }
    if len(labels) > 1:
        return True
    spans = [
        span
        for source in sources
        for span in getattr(source, "spans", [])
    ]
    for index, (left_start, left_end) in enumerate(spans):
        for right_start, right_end in spans[index + 1 :]:
            overlap = max(min(left_end, right_end) - max(left_start, right_start), 0.0)
            if overlap > 0.0:
                return True
    return False
