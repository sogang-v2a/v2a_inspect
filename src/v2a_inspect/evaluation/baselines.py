from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from v2a_inspect.dataset.records import DatasetRecord
from v2a_inspect.evaluation.metrics import structural_metrics


class BaselineResult(BaseModel):
    strategy: Literal["legacy", "tool_only", "agentic", "tta_only", "vta_only"]
    metrics: dict[str, float] = Field(default_factory=dict)


class DownstreamExperimentHook(BaseModel):
    strategy: str
    command: str
    enabled: bool = True


def compare_baselines(
    *,
    reference: DatasetRecord,
    candidates: dict[str, DatasetRecord],
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    for strategy, candidate in candidates.items():
        results.append(
            BaselineResult(strategy=strategy, metrics=structural_metrics(reference, candidate))
        )
    return results


def build_downstream_generation_hooks(record: DatasetRecord) -> list[DownstreamExperimentHook]:
    strategies = ["legacy", "tool_only", "agentic", "tta_only", "vta_only"]
    return [
        DownstreamExperimentHook(
            strategy=strategy,
            command=f"python -m v2a_inspect.cli run-baseline --strategy {strategy} --record {record.record_id}",
        )
        for strategy in strategies
    ]


def crop_evidence_ablation(
    *,
    with_crops: DatasetRecord,
    without_crops: DatasetRecord,
) -> dict[str, float]:
    return {
        "source_coverage_delta": round(
            structural_metrics(with_crops, with_crops)["source_coverage"]
            - structural_metrics(with_crops, without_crops)["source_coverage"],
            4,
        ),
        "generation_group_delta": round(
            len(with_crops.bundle.generation_groups) - len(without_crops.bundle.generation_groups),
            4,
        ),
    }
