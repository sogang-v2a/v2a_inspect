from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from time import perf_counter
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qs, urlparse

from v2a_inspect.contracts.adapters import bundle_to_grouped_analysis
from v2a_inspect.review import persist_bundle
from .settings import get_server_runtime_settings
from v2a_inspect.workflows import InspectOptions, InspectState

from .bootstrap import WeightsBootstrapper, WeightsManifest
from .agentic import run_agentic_tool_loop
from .crops import group_crop_paths_by_track
from .finalize import build_final_bundle, build_interim_bundle
from .gpu_runtime import inspect_nvidia_runtime, runtime_check_to_json
from .model_runtime import clear_cuda_cache
from .telemetry import ensure_runtime_trace_path, record_stage, stage_start
from .tool_registry import build_tool_registry

if TYPE_CHECKING:
    from v2a_inspect.contracts import MultitrackDescriptionBundle
    from .adjudicator import GeminiIssueJudge
    from .embeddings import EmbeddingClient, LabelClient
    from .description_writer import GeminiDescriptionWriter
    from .scene_hypotheses import GeminiSceneHypothesisProposer
    from .sam3 import Sam3Client


class ToolingRuntime:
    def __init__(
        self,
        *,
        bootstrapper: WeightsBootstrapper,
        weights_manifest: WeightsManifest,
        resolved_artifacts: dict[str, Path],
        runtime_profile: Literal["mig10_safe", "full_gpu", "cpu_dev"],
    ) -> None:
        self.bootstrapper = bootstrapper
        self.weights_manifest = weights_manifest
        self.resolved_artifacts = resolved_artifacts
        self.runtime_profile = runtime_profile
        self._sam3_client: "Sam3Client | None" = None
        self._embedding_client: "EmbeddingClient | None" = None
        self._label_client: "LabelClient | None" = None
        self._description_writer: "GeminiDescriptionWriter | None" = None
        self._adjudication_judge: "GeminiIssueJudge | None" = None
        self._scene_hypothesis_proposer: "GeminiSceneHypothesisProposer | None" = None

    @property
    def should_release_clients(self) -> bool:
        return self.runtime_profile == "mig10_safe"

    @property
    def residency_mode(self) -> Literal["resident", "release_after_stage"]:
        return "release_after_stage" if self.should_release_clients else "resident"

    @property
    def sam3_client(self) -> "Sam3Client":
        if self._sam3_client is None:
            from .sam3 import Sam3Client

            self._sam3_client = Sam3Client(model_dir=self.resolved_artifacts["sam3"])
        return self._sam3_client

    @property
    def embedding_client(self) -> "EmbeddingClient":
        if self._embedding_client is None:
            from .embeddings import EmbeddingClient

            self._embedding_client = EmbeddingClient(
                model_dir=self.resolved_artifacts["embedding"]
            )
        return self._embedding_client

    @property
    def label_client(self) -> "LabelClient":
        if self._label_client is None:
            from .embeddings import LabelClient

            self._label_client = LabelClient(model_dir=self.resolved_artifacts["label"])
        return self._label_client

    @property
    def description_writer(self) -> "GeminiDescriptionWriter | None":
        if self._description_writer is None:
            if os.getenv("V2A_DISABLE_DESCRIPTION_WRITER", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return None
            server_settings = get_server_runtime_settings()
            if server_settings.gemini_api_key is None:
                return None
            from .description_writer import GeminiDescriptionWriter

            self._description_writer = GeminiDescriptionWriter(
                api_key=server_settings.gemini_api_key
            )
        return self._description_writer

    @property
    def adjudication_judge(self) -> "GeminiIssueJudge | None":
        if self._adjudication_judge is None:
            if os.getenv("V2A_DISABLE_ADJUDICATION_JUDGE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return None
            server_settings = get_server_runtime_settings()
            if server_settings.gemini_api_key is None:
                return None
            from .adjudicator import GeminiIssueJudge

            self._adjudication_judge = GeminiIssueJudge(
                api_key=server_settings.gemini_api_key
            )
        return self._adjudication_judge

    @property
    def scene_hypothesis_proposer(self) -> "GeminiSceneHypothesisProposer | None":
        if self._scene_hypothesis_proposer is None:
            server_settings = get_server_runtime_settings()
            if server_settings.gemini_api_key is None:
                return None
            from .scene_hypotheses import GeminiSceneHypothesisProposer

            self._scene_hypothesis_proposer = GeminiSceneHypothesisProposer(
                model="gemini-2.5-flash",
                api_key=server_settings.gemini_api_key,
            )
        return self._scene_hypothesis_proposer

    def artifacts_missing(self) -> list[str]:
        return [
            name
            for name, path in self.resolved_artifacts.items()
            if not path.exists()
        ]

    def release_client(
        self, client_name: Literal["sam3", "embedding", "label"]
    ) -> None:
        if client_name == "sam3":
            self._sam3_client = None
        elif client_name == "embedding":
            self._embedding_client = None
        else:
            self._label_client = None
        clear_cuda_cache()

    def release_all(self) -> None:
        self._sam3_client = None
        self._embedding_client = None
        self._label_client = None
        clear_cuda_cache()

    def resident_client_names(self) -> list[str]:
        clients: list[str] = []
        if self._sam3_client is not None:
            clients.append("sam3")
        if self._embedding_client is not None:
            clients.append("embedding")
        if self._label_client is not None:
            clients.append("label")
        if self._description_writer is not None:
            clients.append("description_writer")
        if self._adjudication_judge is not None:
            clients.append("adjudication_judge")
        return clients

    def warmup_visual_clients(self) -> dict[str, object]:
        timings: dict[str, float] = {}
        status: dict[str, str] = {}
        for client_name in ("sam3", "embedding", "label"):
            started = perf_counter()
            try:
                getattr(self, f"{client_name}_client")
                timings[client_name] = round(perf_counter() - started, 4)
                status[client_name] = "ready"
            except Exception as exc:  # noqa: BLE001
                timings[client_name] = round(perf_counter() - started, 4)
                status[client_name] = f"error: {exc}"
                raise
        return {
            "model_load_status": status,
            "model_load_seconds": timings,
            "resident_models": self.resident_client_names(),
        }


@lru_cache(maxsize=1)
def build_tooling_runtime() -> ToolingRuntime:
    server_settings = get_server_runtime_settings()
    bootstrapper = WeightsBootstrapper(
        cache_dir=Path(server_settings.model_cache_dir),
        hf_token=server_settings.hf_token,
    )
    weights_manifest = bootstrapper.load_manifest(
        Path(server_settings.weights_manifest_path)
    )
    if not weights_manifest.artifacts:
        raise FileNotFoundError(
            f"No model artifacts are defined in {server_settings.weights_manifest_path}."
        )
    resolved_artifacts = bootstrapper.resolve_manifest(weights_manifest)
    missing = [
        name for name, path in resolved_artifacts.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing bootstrapped model artifacts: " + ", ".join(sorted(missing))
        )
    return ToolingRuntime(
        bootstrapper=bootstrapper,
        weights_manifest=weights_manifest,
        resolved_artifacts=resolved_artifacts,
        runtime_profile=server_settings.runtime_profile,
    )


def _readyz_payload(
    *, include_model_load_check: bool = False
) -> tuple[dict[str, object], HTTPStatus]:
    server_settings = get_server_runtime_settings()
    gpu_check = inspect_nvidia_runtime(
        minimum_vram_gb=server_settings.minimum_gpu_vram_gb
    )
    payload: dict[str, object] = {
        "runtime_mode": server_settings.runtime_mode,
        "runtime_profile": server_settings.runtime_profile,
        "remote_gpu_target": server_settings.remote_gpu_target,
        "gpu_check": json.loads(runtime_check_to_json(gpu_check)),
    }
    if not gpu_check.available and server_settings.runtime_profile != "cpu_dev":
        payload.update(
            {
                "ok": False,
                "bootstrap_ready": False,
                "tooling_runtime_ready": False,
                "tooling_error": "Remote GPU runtime is unavailable.",
            }
        )
        return payload, HTTPStatus.SERVICE_UNAVAILABLE

    try:
        tooling_runtime = build_tooling_runtime()
        missing = tooling_runtime.artifacts_missing()
        tooling_error = None
        model_load_status: dict[str, str] = {}
        warnings: list[str] = []
        if include_model_load_check:
            warnings.append(
                "The load_models query flag is deprecated; use POST /warmup for persistent model warmup."
            )
        payload.update(
            {
                "ok": not missing and tooling_error is None,
                "bootstrap_ready": not missing,
                "missing_artifacts": missing,
                "tooling_runtime_ready": not missing and tooling_error is None,
                "tooling_error": tooling_error,
                "model_load_status": model_load_status,
                "resident_models": tooling_runtime.resident_client_names(),
                "residency_mode": tooling_runtime.residency_mode,
                "warnings": warnings,
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "ok": False,
                "bootstrap_ready": False,
                "tooling_runtime_ready": False,
                "tooling_error": str(exc),
            }
        )
    status = HTTPStatus.OK if payload["ok"] else HTTPStatus.SERVICE_UNAVAILABLE
    return payload, status


def _warmup_payload(
    *,
    source: Literal["warmup_endpoint", "readyz_alias"] = "warmup_endpoint",
) -> tuple[dict[str, object], HTTPStatus]:
    payload, status = _readyz_payload(include_model_load_check=False)
    if status != HTTPStatus.OK:
        payload["warmup_ok"] = False
        payload["warmup_source"] = source
        return payload, status

    tooling_runtime = build_tooling_runtime()
    try:
        warmup = tooling_runtime.warmup_visual_clients()
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "ok": False,
                "warmup_ok": False,
                "warmup_source": source,
                "tooling_error": str(exc),
                "resident_models": tooling_runtime.resident_client_names(),
                "residency_mode": tooling_runtime.residency_mode,
            }
        )
        return payload, HTTPStatus.INTERNAL_SERVER_ERROR

    warnings = list(payload.get("warnings", []))
    if source == "readyz_alias":
        warnings.append(
            "POST /warmup is the preferred warmup path; /readyz?load_models=true is a deprecated alias."
        )
    payload.update(
        {
            "warmup_ok": True,
            "warmup_source": source,
            "model_load_status": warmup["model_load_status"],
            "model_load_seconds": warmup["model_load_seconds"],
            "resident_models": warmup["resident_models"],
            "residency_mode": tooling_runtime.residency_mode,
            "warnings": warnings,
        }
    )
    return payload, HTTPStatus.OK


def _runtime_info_payload() -> tuple[dict[str, object], HTTPStatus]:
    server_settings = get_server_runtime_settings()
    payload: dict[str, object] = {
        "runtime_mode": server_settings.runtime_mode,
        "runtime_profile": server_settings.runtime_profile,
        "effective_runtime_profile": server_settings.runtime_profile,
        "runtime_profile_source": "server_settings",
        "remote_gpu_target": server_settings.remote_gpu_target,
        "model_cache_dir": str(server_settings.model_cache_dir),
        "weights_manifest_path": str(server_settings.weights_manifest_path),
        "minimum_gpu_vram_gb": server_settings.minimum_gpu_vram_gb,
        "server_bind_host": server_settings.server_bind_host,
        "server_bind_port": server_settings.server_bind_port,
    }
    try:
        tooling_runtime = build_tooling_runtime()
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "tooling_runtime_ready": False,
                "tooling_error": str(exc),
                "residency_mode": "unknown",
                "resident_models": [],
            }
        )
        return payload, HTTPStatus.SERVICE_UNAVAILABLE

    payload.update(
        {
            "tooling_runtime_ready": True,
            "tooling_error": None,
            "residency_mode": tooling_runtime.residency_mode,
            "resident_models": tooling_runtime.resident_client_names(),
        }
    )
    return payload, HTTPStatus.OK


def _analyze_with_pipeline(
    *,
    video_path: str,
    options: InspectOptions,
    tooling_runtime: ToolingRuntime,
) -> InspectState:
    if options.pipeline_mode == "agentic_tool_first":
        return _run_agentic_tool_first_pipeline(
            video_path=video_path,
            options=options,
            tooling_runtime=tooling_runtime,
        )
    if options.pipeline_mode == "tool_first_foundation":
        return _run_tool_first_pipeline(
            video_path=video_path,
            options=options,
            tooling_runtime=tooling_runtime,
        )
    raise ValueError(
        f"Unsupported pipeline_mode {options.pipeline_mode!r}; only tool-first modes remain."
    )


def _run_tool_first_pipeline(
    *,
    video_path: str,
    options: InspectOptions,
    tooling_runtime: ToolingRuntime,
    bundle_mode: Literal["final", "interim"] = "final",
) -> InspectState:
    registry = build_tool_registry(tooling_runtime)
    state: InspectState = {
        "video_path": video_path,
        "options": options,
        "recovery_actions": [],
        "recovery_attempts": [],
        "stage_history": [],
        "effective_runtime_profile": tooling_runtime.runtime_profile,
        "runtime_profile_source": "server_settings",
        "runtime_residency_mode": tooling_runtime.residency_mode,
        "resident_models_before_run": tooling_runtime.resident_client_names(),
        "warm_start": set(("sam3", "embedding", "label")).issubset(
            set(tooling_runtime.resident_client_names())
        ),
    }
    started = stage_start()
    structural = registry["structural_overview"].handler(
        video_path=video_path,
        target_scene_seconds=5.0,
    )
    probe = structural["probe"]
    candidate_cuts = list(structural["candidate_cuts"])
    evidence_windows = list(structural["evidence_windows"])
    frame_batches = list(structural["frame_batches"])
    storyboard_path = str(structural["storyboard_path"])
    artifact_run_dir = str(structural["artifact_root"])
    analysis_video_path = str(structural["analysis_video_path"])
    state["artifact_run_dir"] = artifact_run_dir
    state["analysis_video_path"] = analysis_video_path
    state["storyboard_path"] = storyboard_path
    state["frames_per_window"] = int(structural.get("frames_per_scene", 2))
    ensure_runtime_trace_path(state)
    record_stage(
        state,
        stage="structural_overview",
        started_at=started,
        metrics={
            "candidate_cut_count": len(candidate_cuts),
            "evidence_window_count": len(evidence_windows),
            "sampled_frame_count": sum(len(batch.frames) for batch in frame_batches),
            "frames_per_window": state["frames_per_window"],
        },
    )

    started = stage_start()
    source_hypotheses = registry["propose_source_hypotheses"].handler(
        frame_batches=frame_batches,
        storyboard_path=storyboard_path,
        output_root=artifact_run_dir,
    )
    state["scene_hypotheses_by_window"] = dict(
        source_hypotheses["scene_hypotheses_by_window"]
    )
    state["proposal_provenance_by_window"] = dict(
        source_hypotheses["proposal_provenance_by_window"]
    )
    record_stage(
        state,
        stage="propose_source_hypotheses",
        started_at=started,
        metrics={
            "window_count": len(frame_batches),
            "window_candidate_count": sum(
                len(payload.get("extraction_prompts", []))
                for payload in source_hypotheses["expanded_candidates_by_window"].values()
            ),
            "window_hypothesis_count": len(state["scene_hypotheses_by_window"]),
        },
    )

    started = stage_start()
    verified_hypotheses = registry["verify_scene_hypotheses"].handler(
        frame_batches=frame_batches,
        ontology_scores_by_window=source_hypotheses["ontology_scores_by_window"],
        scene_hypotheses_by_window=source_hypotheses["scene_hypotheses_by_window"],
        moving_region_labels_by_window=source_hypotheses["moving_region_labels_by_window"],
        expanded_candidates_by_window=source_hypotheses["expanded_candidates_by_window"],
    )
    state["verified_hypotheses_by_window"] = dict(
        verified_hypotheses["verified_hypotheses_by_window"]
    )
    state["scene_prompt_candidates"] = dict(verified_hypotheses["prompts_by_scene"])
    state["proposal_provenance_by_window"] = dict(
        verified_hypotheses["proposal_provenance_by_window"]
    )
    record_stage(
        state,
        stage="verify_scene_hypotheses",
        started_at=started,
        metrics={
            "verified_window_count": len(state["verified_hypotheses_by_window"]),
            "verified_prompt_count": sum(
                len(prompts)
                for prompts in state["scene_prompt_candidates"].values()
            ),
            "uncertain_hypothesis_count": sum(
                len(payload.get("uncertain_hypotheses", []))
                for payload in state["verified_hypotheses_by_window"].values()
            ),
        },
    )

    started = stage_start()
    extraction = registry["extract_entities"].handler(
        frame_batches=frame_batches,
        prompts_by_scene=dict(state["scene_prompt_candidates"]),
        storyboard_path=storyboard_path,
        output_root=artifact_run_dir,
    )
    tracks = _coerce_tracks(extraction)
    record_stage(
        state,
        stage="extract_entities",
        started_at=started,
        metrics={
            "track_count": len(tracks),
            "extraction_strategy": getattr(extraction, "strategy", None),
        },
    )
    if tooling_runtime.should_release_clients:
        tooling_runtime.release_client("sam3")
    started = stage_start()
    track_crops = registry["crop_tracks"].handler(
        frame_batches=frame_batches,
        tracks=tracks,
        output_dir=str(Path(storyboard_path).parent / "crops"),
    )
    track_image_paths = group_crop_paths_by_track(track_crops)
    record_stage(
        state,
        stage="crop_tracks",
        started_at=started,
        metrics={
            "crop_count": len(track_crops),
            "tracks_with_crops": len(track_image_paths),
        },
    )
    started = stage_start()
    embeddings = (
        registry["embed_track_crops"].handler(track_image_paths=track_image_paths)
        if track_image_paths
        else []
    )
    candidate_groups = list(
        getattr(
            registry["group_embeddings"].handler(
                embeddings=embeddings,
                tracks_by_id={track.track_id: track for track in tracks},
            ),
            "groups",
            [],
        )
    )
    record_stage(
        state,
        stage="embed_track_crops",
        started_at=started,
        metrics={
            "embedding_count": len(embeddings),
            "candidate_group_count": len(candidate_groups),
        },
    )
    if tooling_runtime.should_release_clients:
        tooling_runtime.release_client("embedding")
    started = stage_start()
    track_label_candidates = (
        registry["score_track_labels"].handler(track_image_paths=track_image_paths)
        if track_image_paths
        else {}
    )
    routing_decisions = (
        dict(registry["routing_priors"].handler(tracks=tracks)) if tracks else {}
    )
    record_stage(
        state,
        stage="score_track_labels",
        started_at=started,
        metrics={
            "labeled_track_count": len(track_label_candidates),
            "routing_track_count": len(routing_decisions),
        },
    )
    if tooling_runtime.should_release_clients:
        tooling_runtime.release_client("label")
    started = stage_start()
    refined_structure = registry["refine_candidate_cuts"].handler(
        probe=probe,
        candidate_cuts=candidate_cuts,
        frame_batches=frame_batches,
        tracks=tracks,
        label_candidates_by_track=track_label_candidates,
        storyboard_path=storyboard_path,
    )
    candidate_cuts = list(refined_structure["candidate_cuts"])
    evidence_windows = list(refined_structure["evidence_windows"])
    record_stage(
        state,
        stage="refine_candidate_cuts",
        started_at=started,
        metrics={
            "candidate_cut_count": len(candidate_cuts),
            "evidence_window_count": len(evidence_windows),
        },
    )
    started = stage_start()
    semantics = registry["build_source_semantics"].handler(
        tracks=tracks,
        embeddings=embeddings,
        track_crops=track_crops,
        label_candidates_by_track=track_label_candidates,
        evidence_windows=evidence_windows,
        candidate_groups=candidate_groups,
        routing_decisions_by_track=routing_decisions,
        scene_hypotheses_by_window=dict(state.get("scene_hypotheses_by_window", {})),
        proposal_provenance_by_window=dict(state.get("proposal_provenance_by_window", {})),
    )
    record_stage(
        state,
        stage="build_source_semantics",
        started_at=started,
        metrics={
            "identity_edge_count": len(semantics["identity_edges"]),
            "physical_source_count": len(semantics["physical_sources"]),
            "sound_event_count": len(semantics["sound_events"]),
            "ambience_bed_count": len(semantics["ambience_beds"]),
            "generation_group_count": len(semantics["generation_groups"]),
            "recipe_signature_count": len(semantics.get("recipe_signatures", {})),
            "recipe_grouping_seconds": semantics.get("recipe_grouping_seconds"),
        },
    )

    state.update({
        "artifact_run_dir": artifact_run_dir,
        "video_probe": probe,
        "candidate_cuts": candidate_cuts,
        "evidence_windows": evidence_windows,
        "frame_batches": frame_batches,
        "analysis_video_path": analysis_video_path,
        "storyboard_path": storyboard_path,
        "sam3_track_set": extraction,
        "scene_prompt_candidates": dict(state.get("scene_prompt_candidates", {})),
        "scene_hypotheses_by_window": dict(state.get("scene_hypotheses_by_window", {})),
        "proposal_provenance_by_window": dict(state.get("proposal_provenance_by_window", {})),
        "verified_hypotheses_by_window": dict(state.get("verified_hypotheses_by_window", {})),
        "track_crops": track_crops,
        "entity_embeddings": embeddings,
        "candidate_groups": candidate_groups,
        "track_routing_decisions": routing_decisions,
        "track_label_candidates": track_label_candidates,
        "physical_sources": list(semantics["physical_sources"]),
        "sound_event_segments": list(semantics["sound_events"]),
        "ambience_beds": list(semantics["ambience_beds"]),
        "generation_groups": list(semantics["generation_groups"]),
        "recipe_signatures_by_group": {
            key: value.model_dump(mode="json")
            for key, value in semantics.get("recipe_signatures", {}).items()
        },
        "identity_edges": list(semantics["identity_edges"]),
        "warnings": [],
        "errors": [],
        "progress_messages": [
            f"Tool-first pipeline: proposed {len(candidate_cuts)} candidate cuts.",
            f"Tool-first pipeline: built {len(evidence_windows)} evidence windows.",
            f"Tool-first pipeline: sampled {sum(len(batch.frames) for batch in frame_batches)} frames.",
            f"Tool-first pipeline: proposed {sum(len(prompts) for prompts in state.get('scene_prompt_candidates', {}).values())} extraction prompts.",
            f"Tool-first pipeline: extracted {len(tracks)} source tracks.",
            f"Tool-first pipeline: generated {len(track_crops)} track crops.",
            f"Tool-first pipeline: embedded {len(embeddings)} crop-backed track identities.",
            f"Tool-first pipeline: refined structure with {len(candidate_cuts)} merged candidate cuts.",
            "Tool-first pipeline: built source, event, ambience, and generation-group semantics.",
        ],
    })
    bundle = (
        build_final_bundle(
            state,
            description_writer=getattr(tooling_runtime, "description_writer", None),
        )
        if bundle_mode == "final"
        else build_interim_bundle(state)
    )
    bundle.pipeline_metadata["resident_models_after_run"] = tooling_runtime.resident_client_names()
    started = stage_start()
    _persist_runtime_bundle(bundle, state)
    record_stage(
        state,
        stage="persist_bundle",
        started_at=started,
        metrics={"bundle_path": state.get("bundle_path"), "pipeline_mode": options.pipeline_mode},
    )
    grouped = bundle_to_grouped_analysis(bundle)
    state["scene_analysis"] = grouped.scene_analysis
    state["grouped_analysis"] = grouped
    state["multitrack_bundle"] = bundle
    return state


def _run_agentic_tool_first_pipeline(
    *,
    video_path: str,
    options: InspectOptions,
    tooling_runtime: ToolingRuntime,
) -> InspectState:
    state = _run_tool_first_pipeline(
        video_path=video_path,
        options=options.model_copy(update={"pipeline_mode": "tool_first_foundation"}),
        tooling_runtime=tooling_runtime,
        bundle_mode="interim",
    )
    state["options"] = options
    state, planner_state, trace_path = run_agentic_tool_loop(
        inspect_state=state,
        tooling_runtime=tooling_runtime,
    )
    state["agent_trace_path"] = trace_path
    bundle = state["multitrack_bundle"]
    bundle.artifacts.trace_path = trace_path
    bundle.pipeline_metadata["agent_review_trace_path"] = trace_path
    bundle.pipeline_metadata["agent_review_issue_count"] = len(planner_state.issues)
    bundle.pipeline_metadata["agent_review_tool_calls"] = len(planner_state.tool_calls)
    _persist_runtime_bundle(bundle, state)
    grouped = bundle_to_grouped_analysis(bundle)
    state["scene_analysis"] = grouped.scene_analysis
    state["grouped_analysis"] = grouped
    state["multitrack_bundle"] = bundle
    return state


def _coerce_tracks(extraction_result: object) -> list[object]:
    if isinstance(extraction_result, dict):
        tracks = extraction_result.get("tracks", [])
    else:
        tracks = getattr(extraction_result, "tracks", [])
    return list(tracks) if isinstance(tracks, list) else list(tracks or [])


def _persist_runtime_bundle(
    bundle: "MultitrackDescriptionBundle", state: InspectState
) -> None:
    artifact_run_dir = state.get("artifact_run_dir")
    if not isinstance(artifact_run_dir, str) or not artifact_run_dir:
        return
    bundle_path = Path(artifact_run_dir) / "bundle.json"
    persist_bundle(bundle, bundle_path)
    state["bundle_path"] = str(bundle_path)
    bundle.artifacts.bundle_path = str(bundle_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v2a-inspect-server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("runtime-info", help="Show server runtime configuration")
    subparsers.add_parser(
        "bootstrap", help="Bootstrap model weights into the configured cache"
    )
    subparsers.add_parser(
        "check", help="Validate NVIDIA GPU visibility and minimum VRAM"
    )
    subparsers.add_parser("warmup", help="Load visual models and keep them resident")
    subparsers.add_parser("serve", help="Run the server runtime HTTP API")

    args = parser.parse_args(argv)
    if args.command == "runtime-info":
        return _run_runtime_info()
    if args.command == "bootstrap":
        return _run_bootstrap()
    if args.command == "serve":
        return _run_serve()
    if args.command == "warmup":
        return _run_warmup()
    return _run_check()


def _run_runtime_info() -> int:
    payload, _ = _runtime_info_payload()
    print(json.dumps(payload, indent=2))
    return 0


def _run_warmup() -> int:
    payload, status = _warmup_payload()
    print(json.dumps(payload, indent=2))
    return 0 if status == HTTPStatus.OK else 1


def _run_bootstrap() -> int:
    server_settings = get_server_runtime_settings()
    bootstrapper = WeightsBootstrapper(
        cache_dir=Path(server_settings.model_cache_dir),
        hf_token=server_settings.hf_token,
    )
    manifest = bootstrapper.load_manifest(Path(server_settings.weights_manifest_path))
    resolved = bootstrapper.ensure_manifest(manifest)
    print(json.dumps({name: str(path) for name, path in resolved.items()}, indent=2))
    return 0


def _run_check() -> int:
    result = inspect_nvidia_runtime(
        minimum_vram_gb=get_server_runtime_settings().minimum_gpu_vram_gb
    )
    print(runtime_check_to_json(result))
    return 0 if result.available else 1


def _run_serve() -> int:
    server_settings = get_server_runtime_settings()
    server = ThreadingHTTPServer(
        (server_settings.server_bind_host, server_settings.server_bind_port),
        _build_handler(),
    )
    print(
        json.dumps(
            {
                "message": "v2a-inspect-server listening",
                "host": server_settings.server_bind_host,
                "port": server_settings.server_bind_port,
            }
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _build_handler() -> type[BaseHTTPRequestHandler]:
    class RuntimeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                server_settings = get_server_runtime_settings()
                self._write_json(
                    {
                        "ok": True,
                        "runtime_mode": server_settings.runtime_mode,
                        "runtime_profile": server_settings.runtime_profile,
                        "remote_gpu_target": server_settings.remote_gpu_target,
                    }
                )
                return
            if parsed.path in {"/readyz", "/health"}:
                query = parse_qs(parsed.query)
                include_model_load_check = query.get("load_models", ["0"])[0] in {
                    "1",
                    "true",
                    "yes",
                }
                if include_model_load_check:
                    payload, status = _warmup_payload(source="readyz_alias")
                else:
                    payload, status = _readyz_payload(
                        include_model_load_check=False
                    )
                self._write_json(payload, status_code=status)
                return
            if parsed.path == "/runtime-info":
                payload, status = _runtime_info_payload()
                self._write_json(payload, status_code=status)
                return
            if parsed.path == "/warmup":
                self.send_error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "Use POST /warmup for persistent model warmup.",
                )
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.startswith("/upload"):
                filename = self.headers.get("X-Filename", "video.mp4")
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    self.send_error(
                        HTTPStatus.BAD_REQUEST, "Upload body must not be empty."
                    )
                    return
                upload_bytes = self.rfile.read(content_length)
                upload_path = _write_uploaded_video(
                    filename=filename,
                    raw_bytes=upload_bytes,
                )
                self._write_json({"ok": True, "video_path": upload_path})
                return
            if self.path == "/bootstrap":
                try:
                    server_settings = get_server_runtime_settings()
                    bootstrapper = WeightsBootstrapper(
                        cache_dir=Path(server_settings.model_cache_dir),
                        hf_token=server_settings.hf_token,
                    )
                    manifest = bootstrapper.load_manifest(
                        Path(server_settings.weights_manifest_path)
                    )
                    if not manifest.artifacts:
                        raise FileNotFoundError(
                            f"No model artifacts are defined in {server_settings.weights_manifest_path}."
                        )
                    resolved = bootstrapper.ensure_manifest(manifest)
                    build_tooling_runtime.cache_clear()
                    self._write_json(
                        {
                            "ok": True,
                            "artifacts": {
                                name: str(path) for name, path in resolved.items()
                            },
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    self._write_json(
                        {
                            "ok": False,
                            "error": str(exc),
                        },
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return
            if self.path == "/warmup":
                try:
                    payload, status = _warmup_payload()
                    self._write_json(payload, status_code=status)
                except Exception as exc:  # noqa: BLE001
                    self._write_json(
                        {
                            "ok": False,
                            "warmup_ok": False,
                            "error": str(exc),
                        },
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return
            if self.path == "/analyze":
                try:
                    payload = self._read_json()
                    video_path = payload.get("video_path")
                    video_filename = payload.get("video_filename")
                    options_payload = payload.get("options", {})
                    if not isinstance(video_path, str):
                        self.send_error(
                            HTTPStatus.BAD_REQUEST,
                            "video_path is required",
                        )
                        return
                    if not isinstance(options_payload, dict):
                        self.send_error(
                            HTTPStatus.BAD_REQUEST, "options must be an object"
                        )
                        return

                    options = InspectOptions.model_validate(options_payload)
                    resolved_video_path = _resolve_request_video_path(
                        video_path=video_path if isinstance(video_path, str) else "",
                        video_filename=video_filename
                        if isinstance(video_filename, str)
                        else "video.mp4",
                    )
                    gpu_check = inspect_nvidia_runtime(
                        minimum_vram_gb=get_server_runtime_settings().minimum_gpu_vram_gb
                    )
                    if not gpu_check.available:
                        self._write_json(
                            {
                                "ok": False,
                                "error": "Remote GPU runtime is unavailable for /analyze.",
                                "gpu_check": json.loads(
                                    runtime_check_to_json(gpu_check)
                                ),
                            },
                            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                        )
                        return
                    tooling_runtime = build_tooling_runtime()
                    server_options = options.model_copy(
                        update={
                            "runtime_mode": "in_process",
                            "runtime_profile": tooling_runtime.runtime_profile,
                        }
                    )
                    runtime_profile_warning: str | None = None
                    if options.runtime_profile != tooling_runtime.runtime_profile:
                        runtime_profile_warning = (
                            "Requested runtime_profile "
                            f"{options.runtime_profile!r} was ignored; "
                            f"server is using {tooling_runtime.runtime_profile!r}."
                        )
                    state = _analyze_with_pipeline(
                        video_path=resolved_video_path,
                        options=server_options,
                        tooling_runtime=tooling_runtime,
                    )
                    bundle = state.get("multitrack_bundle") or build_final_bundle(
                        state,
                        description_writer=getattr(
                            tooling_runtime, "description_writer", None
                        ),
                    )
                    state["multitrack_bundle"] = bundle
                    warnings = list(state.get("warnings", []))
                    if runtime_profile_warning is not None:
                        warnings.append(runtime_profile_warning)
                    bundle.pipeline_metadata["effective_runtime_profile"] = tooling_runtime.runtime_profile
                    bundle.pipeline_metadata["runtime_profile_source"] = "server_settings"
                    bundle.pipeline_metadata["runtime_residency_mode"] = tooling_runtime.residency_mode
                    bundle.pipeline_metadata["warm_start"] = state.get("warm_start", False)
                    bundle.pipeline_metadata["resident_models_before_run"] = list(
                        state.get("resident_models_before_run", [])
                    )
                    bundle.pipeline_metadata["resident_models_after_run"] = tooling_runtime.resident_client_names()
                    self._write_json(
                        {
                            "multitrack_bundle": bundle.model_dump(mode="json"),
                            "warnings": warnings,
                            "progress_messages": state.get("progress_messages", []),
                            "effective_runtime_profile": tooling_runtime.runtime_profile,
                            "runtime_profile_source": "server_settings",
                            "residency_mode": tooling_runtime.residency_mode,
                            "warm_start": state.get("warm_start", False),
                            "resident_models_before_run": list(
                                state.get("resident_models_before_run", [])
                            ),
                            "resident_models_after_run": tooling_runtime.resident_client_names(),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    self._write_json(
                        {
                            "ok": False,
                            "error": str(exc),
                        },
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return None

        def _read_json(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(body) if body else {}
            if not isinstance(payload, dict):
                raise TypeError("Request payload must be a JSON object.")
            return payload

        def _write_json(
            self,
            payload: Mapping[str, object],
            *,
            status_code: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RuntimeHandler


def _resolve_request_video_path(
    *,
    video_path: str,
    video_filename: str,
) -> str:
    del video_filename
    if not video_path:
        raise ValueError("video_path is required.")
    candidate = Path(video_path)
    if not candidate.exists():
        raise ValueError("video_path must reference an existing uploaded file.")
    allowed_root = (
        get_server_runtime_settings().shared_video_dir or Path(tempfile.gettempdir())
    )
    resolved_candidate = candidate.resolve()
    resolved_root = Path(allowed_root).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "video_path must point to a file created by POST /upload inside the managed upload directory."
        ) from exc
    return str(resolved_candidate)


def _write_uploaded_video(*, filename: str, raw_bytes: bytes) -> str:
    target_dir = _prepare_upload_dir()
    target_path = target_dir / _sanitize_filename(filename)
    target_path.write_bytes(raw_bytes)
    return str(target_path)


def _prepare_upload_dir() -> Path:
    target_root = (
        get_server_runtime_settings().shared_video_dir or Path(tempfile.gettempdir())
    )
    resolved_root = Path(target_root)
    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        resolved_root = Path(tempfile.gettempdir())
        resolved_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix="v2a_inspect_server_upload_",
            dir=str(resolved_root),
        )
    )


def _sanitize_filename(filename: str) -> str:
    return (
        "".join(
            char for char in Path(filename).name if char.isalnum() or char in "._-"
        )
        or "video.mp4"
    )
