from __future__ import annotations

from v2a_inspect.dataset.records import DatasetRecord


def source_coverage_score(record: DatasetRecord) -> float:
    if not record.bundle.evidence_windows:
        return 0.0
    covered = 0.0
    total = 0.0
    for window in record.bundle.evidence_windows:
        window_duration = max(window.end_time - window.start_time, 0.0)
        total += window_duration
        for source in record.bundle.physical_sources:
            for start_time, end_time in source.spans:
                covered += max(
                    min(end_time, window.end_time) - max(start_time, window.start_time),
                    0.0,
                )
    return round(min(covered / total, 1.0), 4) if total else 0.0


def route_agreement(reference: DatasetRecord, candidate: DatasetRecord) -> float:
    reference_routes = {
        group.group_id: group.route_decision.model_type
        for group in reference.bundle.generation_groups
    }
    candidate_routes = {
        group.group_id: group.route_decision.model_type
        for group in candidate.bundle.generation_groups
    }
    if not reference_routes:
        return 1.0
    matches = sum(
        1
        for group_id, route in reference_routes.items()
        if candidate_routes.get(group_id) == route
    )
    return round(matches / len(reference_routes), 4)


def structural_metrics(reference: DatasetRecord, candidate: DatasetRecord) -> dict[str, float]:
    return {
        "source_count_delta": float(
            abs(len(reference.bundle.physical_sources) - len(candidate.bundle.physical_sources))
        ),
        "event_count_delta": float(
            abs(len(reference.bundle.sound_events) - len(candidate.bundle.sound_events))
        ),
        "generation_group_delta": float(
            abs(len(reference.bundle.generation_groups) - len(candidate.bundle.generation_groups))
        ),
        "route_agreement": route_agreement(reference, candidate),
        "source_coverage": source_coverage_score(candidate),
    }
