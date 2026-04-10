from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

from v2a_inspect.contracts import CandidateCut, CutReason, EvidenceWindow, LabelCandidate

from .types import (
    FrameBatch,
    SampledFrame,
    Sam3EntityTrack,
    SceneBoundary,
    VideoProbe,
)

_DEFAULT_ANALYSIS_FPS = 2.0
_DEFAULT_HARD_CUT_THRESHOLD = 0.35
_DEFAULT_SOFT_CUT_THRESHOLD = 0.18
_DEFAULT_MOTION_DELTA_THRESHOLD = 0.12
_DEFAULT_MINIMUM_CUT_SPACING = 0.75
_DEFAULT_MINIMUM_WINDOW_SECONDS = 1.0


def probe_video(video_path: str) -> VideoProbe:
    command = [
        _media_binary("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    format_payload = payload.get("format", {})
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        {},
    )
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    avg_frame_rate = video_stream.get("avg_frame_rate")
    fps = _parse_frame_rate(avg_frame_rate)
    duration = float(format_payload.get("duration") or 0.0)
    frame_count = (
        int(video_stream["nb_frames"])
        if str(video_stream.get("nb_frames", "")).isdigit()
        else None
    )
    return VideoProbe(
        video_path=video_path,
        duration_seconds=duration,
        fps=fps,
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        frame_count=frame_count,
        codec_name=video_stream.get("codec_name"),
        format_name=format_payload.get("format_name"),
        has_audio_stream=has_audio,
    )


def build_candidate_cuts(
    video_path: str,
    *,
    probe: VideoProbe | None = None,
    target_scene_seconds: float = 5.0,
    analysis_fps: float = _DEFAULT_ANALYSIS_FPS,
    hard_cut_threshold: float = _DEFAULT_HARD_CUT_THRESHOLD,
    soft_cut_threshold: float = _DEFAULT_SOFT_CUT_THRESHOLD,
    minimum_spacing_seconds: float = _DEFAULT_MINIMUM_CUT_SPACING,
) -> list[CandidateCut]:
    resolved_probe = probe or probe_video(video_path)
    if resolved_probe.duration_seconds <= 0.0:
        return []

    proposals: list[CandidateCut] = []
    previous_difference: float | None = None
    with tempfile.TemporaryDirectory(prefix=f"{Path(video_path).stem}_analysis_") as tmp_dir:
        analysis_frames = _extract_analysis_frames(
            video_path,
            probe=resolved_probe,
            output_dir=Path(tmp_dir),
            analysis_fps=analysis_fps,
        )
        for previous_frame, current_frame in zip(
            analysis_frames, analysis_frames[1:], strict=False
        ):
            difference = _frame_difference(
                previous_frame.image_path,
                current_frame.image_path,
            )
            reason_kind: str | None = None
            if difference >= hard_cut_threshold:
                reason_kind = "shot_boundary"
            elif difference >= soft_cut_threshold:
                reason_kind = "composition_change"
            elif (
                previous_difference is not None
                and abs(difference - previous_difference)
                >= _DEFAULT_MOTION_DELTA_THRESHOLD
                and difference >= (soft_cut_threshold * 0.75)
            ):
                reason_kind = "motion_regime_change"

            if reason_kind is not None:
                proposals.append(
                    CandidateCut(
                        cut_id=f"proposal-{len(proposals)}",
                        timestamp_seconds=round(current_frame.timestamp_seconds, 3),
                        confidence=round(min(max(difference, 0.0), 1.0), 3),
                        reasons=[
                            CutReason(
                                kind=reason_kind,
                                confidence=round(min(max(difference, 0.0), 1.0), 3),
                                rationale=f"frame_difference={difference:.3f}",
                            )
                        ],
                    )
                )
            previous_difference = difference

    proposals.extend(
        _fallback_candidate_cuts(
            duration_seconds=resolved_probe.duration_seconds,
            target_scene_seconds=target_scene_seconds,
        )
    )
    return merge_candidate_cuts(
        proposals,
        minimum_spacing_seconds=minimum_spacing_seconds,
    )


def merge_candidate_cuts(
    candidate_cuts: list[CandidateCut],
    *,
    minimum_spacing_seconds: float = _DEFAULT_MINIMUM_CUT_SPACING,
) -> list[CandidateCut]:
    if not candidate_cuts:
        return []
    ordered = sorted(candidate_cuts, key=lambda cut: cut.timestamp_seconds)
    merged: list[CandidateCut] = []
    current_group: list[CandidateCut] = [ordered[0]]

    for candidate in ordered[1:]:
        if (
            candidate.timestamp_seconds - current_group[-1].timestamp_seconds
            <= minimum_spacing_seconds
        ):
            current_group.append(candidate)
            continue
        merged.append(_merge_cut_group(current_group, len(merged)))
        current_group = [candidate]
    merged.append(_merge_cut_group(current_group, len(merged)))
    return merged


def build_evidence_windows(
    *,
    probe: VideoProbe,
    candidate_cuts: list[CandidateCut],
    minimum_window_seconds: float = _DEFAULT_MINIMUM_WINDOW_SECONDS,
) -> list[EvidenceWindow]:
    accepted_cuts: list[CandidateCut] = []
    last_boundary = 0.0
    for cut in sorted(candidate_cuts, key=lambda item: item.timestamp_seconds):
        if cut.timestamp_seconds <= 0.0 or cut.timestamp_seconds >= probe.duration_seconds:
            continue
        if cut.timestamp_seconds - last_boundary < minimum_window_seconds:
            continue
        if probe.duration_seconds - cut.timestamp_seconds < minimum_window_seconds * 0.5:
            continue
        accepted_cuts.append(cut)
        last_boundary = cut.timestamp_seconds

    boundaries = [0.0, *[cut.timestamp_seconds for cut in accepted_cuts], probe.duration_seconds]
    windows: list[EvidenceWindow] = []
    for index, (start_time, end_time) in enumerate(
        zip(boundaries, boundaries[1:], strict=False)
    ):
        previous_cut = accepted_cuts[index - 1] if index > 0 else None
        next_cut = accepted_cuts[index] if index < len(accepted_cuts) else None
        cut_refs = [
            cut.cut_id
            for cut in (previous_cut, next_cut)
            if cut is not None
        ]
        reason_labels = [
            reason.kind
            for cut in (previous_cut, next_cut)
            if cut is not None
            for reason in cut.reasons
        ]
        rationale = (
            f"window created from {', '.join(reason_labels)}"
            if reason_labels
            else "window created from fallback coverage"
        )
        windows.append(
            EvidenceWindow(
                window_id=f"window-{index:04d}",
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                cut_refs=cut_refs,
                rationale=rationale,
            )
        )
    return windows


def evidence_windows_to_scene_boundaries(
    evidence_windows: list[EvidenceWindow],
    *,
    candidate_cuts: list[CandidateCut] | None = None,
) -> list[SceneBoundary]:
    cut_by_id = {
        candidate.cut_id: candidate
        for candidate in candidate_cuts or []
    }
    boundaries: list[SceneBoundary] = []
    for index, window in enumerate(evidence_windows):
        reason_kinds = {
            reason.kind
            for cut_ref in window.cut_refs
            for reason in cut_by_id.get(cut_ref, CandidateCut(cut_id=cut_ref, timestamp_seconds=window.start_time, confidence=0.0)).reasons
        }
        strategy = (
            "ffmpeg_scene_detect"
            if any(kind != "fallback_window" for kind in reason_kinds)
            else "fixed_window"
        )
        boundaries.append(
            SceneBoundary(
                scene_index=index,
                start_seconds=window.start_time,
                end_seconds=window.end_time,
                strategy=strategy,
            )
        )
    return boundaries


def hydrate_evidence_windows(
    evidence_windows: list[EvidenceWindow],
    frame_batches: list[FrameBatch],
    *,
    storyboard_path: str | None = None,
    clip_paths_by_window: dict[str, str] | None = None,
) -> list[EvidenceWindow]:
    hydrated: list[EvidenceWindow] = []
    all_frames = [frame for batch in frame_batches for frame in batch.frames]
    for window in evidence_windows:
        frame_ids = [
            frame.image_path
            for frame in all_frames
            if window.start_time <= frame.timestamp_seconds <= window.end_time
        ]
        artifact_refs = list(window.artifact_refs)
        if storyboard_path is not None:
            artifact_refs.append(storyboard_path)
        if clip_paths_by_window and window.window_id in clip_paths_by_window:
            artifact_refs.append(clip_paths_by_window[window.window_id])
        hydrated.append(
            window.model_copy(
                update={
                    "sampled_frame_ids": frame_ids,
                    "artifact_refs": artifact_refs,
                }
            )
        )
    return hydrated


def build_context_candidate_cuts(
    *,
    candidate_cuts: list[CandidateCut],
    probe: VideoProbe,
    frame_batches: list[FrameBatch],
    tracks: list[Sam3EntityTrack],
    label_candidates_by_track: dict[str, list[LabelCandidate]],
    storyboard_path: str | None = None,
) -> tuple[list[CandidateCut], list[EvidenceWindow]]:
    merged_candidate_cuts = merge_candidate_cuts(
        [
            *candidate_cuts,
            *_source_lifecycle_cuts(probe=probe, tracks=tracks),
            *_label_context_cuts(
                frame_batches=frame_batches,
                tracks=tracks,
                label_candidates_by_track=label_candidates_by_track,
            ),
            *_interaction_onset_cuts(probe=probe, tracks=tracks),
        ]
    )
    evidence_windows = build_evidence_windows(
        probe=probe,
        candidate_cuts=merged_candidate_cuts,
    )
    return merged_candidate_cuts, hydrate_evidence_windows(
        evidence_windows,
        frame_batches,
        storyboard_path=storyboard_path,
    )


def detect_scenes(
    video_path: str,
    *,
    probe: VideoProbe | None = None,
    target_scene_seconds: float = 5.0,
) -> list[SceneBoundary]:
    resolved_probe = probe or probe_video(video_path)
    if resolved_probe.duration_seconds <= 0.0:
        return [SceneBoundary(scene_index=0, start_seconds=0.0, end_seconds=0.0)]

    candidate_cuts = build_candidate_cuts(
        video_path,
        probe=resolved_probe,
        target_scene_seconds=target_scene_seconds,
    )
    evidence_windows = build_evidence_windows(
        probe=resolved_probe,
        candidate_cuts=candidate_cuts,
    )
    return evidence_windows_to_scene_boundaries(
        evidence_windows,
        candidate_cuts=candidate_cuts,
    )


def sample_frames(
    video_path: str,
    scenes: list[SceneBoundary],
    *,
    output_dir: str,
    frames_per_scene: int = 3,
) -> list[FrameBatch]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    frame_batches: list[FrameBatch] = []
    for scene in scenes:
        timestamps = _scene_sample_timestamps(scene, frames_per_scene=frames_per_scene)
        frames: list[SampledFrame] = []
        for frame_index, timestamp in enumerate(timestamps):
            frame_path = (
                output_root
                / f"scene_{scene.scene_index:04d}_frame_{frame_index:02d}.jpg"
            )
            _extract_single_frame(video_path, timestamp, frame_path)
            frames.append(
                SampledFrame(
                    scene_index=scene.scene_index,
                    timestamp_seconds=timestamp,
                    image_path=str(frame_path),
                )
            )
        frame_batches.append(FrameBatch(scene_index=scene.scene_index, frames=frames))
    return frame_batches


def generate_storyboard(
    frame_batches: list[FrameBatch],
    *,
    output_path: str,
    columns: int = 3,
    tile_size: tuple[int, int] = (220, 124),
) -> str:
    all_frames = [frame for batch in frame_batches for frame in batch.frames]
    if not all_frames:
        raise ValueError("Storyboard requires at least one sampled frame.")

    width, height = tile_size
    rows = (len(all_frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + 24)), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for index, frame in enumerate(all_frames):
        with Image.open(frame.image_path) as image:
            tile = ImageOps.contain(image.convert("RGB"), tile_size)
            x = (index % columns) * width
            y = (index // columns) * (height + 24)
            canvas.paste(tile, (x, y))
            draw.text(
                (x + 6, y + height + 4),
                f"scene {frame.scene_index} @ {frame.timestamp_seconds:.2f}s",
                fill=(235, 235, 235),
            )

    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(resolved_path, format="JPEG")
    return str(resolved_path)


def export_window_clips(
    video_path: str,
    evidence_windows: list[EvidenceWindow],
    *,
    output_dir: str,
    max_windows: int | None = None,
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    clip_paths: dict[str, str] = {}
    selected_windows = evidence_windows[:max_windows] if max_windows is not None else evidence_windows
    for window in selected_windows:
        duration = max(window.end_time - window.start_time, 0.0)
        if duration <= 0.0:
            continue
        clip_path = output_root / f"{window.window_id}.mp4"
        command = [
            _media_binary("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{window.start_time:.3f}",
            "-i",
            video_path,
            "-t",
            f"{duration:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        clip_paths[window.window_id] = str(clip_path)
    return clip_paths


def _extract_single_frame(
    video_path: str, timestamp_seconds: float, output_path: Path
) -> None:
    primary_command = [
        _media_binary("ffmpeg"),
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        str(output_path),
    ]
    try:
        subprocess.run(
            primary_command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        fallback_command = [
            _media_binary("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-ss",
            f"{timestamp_seconds:.3f}",
            "-frames:v",
            "1",
            str(output_path),
        ]
        try:
            subprocess.run(
                fallback_command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            final_fallback_command = [
                _media_binary("ffmpeg"),
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-frames:v",
                "1",
                str(output_path),
            ]
            subprocess.run(
                final_fallback_command,
                check=True,
                capture_output=True,
                text=True,
            )


def _extract_analysis_frames(
    video_path: str,
    *,
    probe: VideoProbe,
    output_dir: Path,
    analysis_fps: float,
) -> list[SampledFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_fps = max(0.5, min(analysis_fps, probe.fps or analysis_fps, 4.0))
    pattern = output_dir / "analysis_%06d.jpg"
    command = [
        _media_binary("ffmpeg"),
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={effective_fps}",
        str(pattern),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    frame_paths = sorted(output_dir.glob("analysis_*.jpg"))
    if not frame_paths:
        fallback_path = output_dir / "analysis_000001.jpg"
        _extract_single_frame(video_path, 0.0, fallback_path)
        frame_paths = [fallback_path]
    return [
        SampledFrame(
            scene_index=0,
            timestamp_seconds=round(min(index / effective_fps, probe.duration_seconds), 3),
            image_path=str(path),
        )
        for index, path in enumerate(frame_paths)
    ]


def _media_binary(name: str) -> str:
    env_value = os.getenv(f"V2A_{name.upper()}_BIN")
    if env_value:
        return env_value
    discovered = shutil.which(name)
    if discovered:
        return discovered
    interpreter_bin = Path(sys.executable).resolve().parent / name
    if interpreter_bin.exists():
        return str(interpreter_bin)
    return name


def _frame_difference(previous_image_path: str, current_image_path: str) -> float:
    with Image.open(previous_image_path) as previous_image, Image.open(
        current_image_path
    ) as current_image:
        previous_gray = ImageOps.grayscale(previous_image.resize((64, 64)))
        current_gray = ImageOps.grayscale(current_image.resize((64, 64)))
        difference = ImageChops.difference(previous_gray, current_gray)
        return float(ImageStat.Stat(difference).mean[0] / 255.0)


def _fallback_candidate_cuts(
    *,
    duration_seconds: float,
    target_scene_seconds: float,
) -> list[CandidateCut]:
    if target_scene_seconds <= 0:
        return []
    cuts: list[CandidateCut] = []
    timestamp = target_scene_seconds
    while timestamp < duration_seconds:
        cuts.append(
            CandidateCut(
                cut_id=f"fallback-{len(cuts)}",
                timestamp_seconds=round(timestamp, 3),
                confidence=0.2,
                reasons=[
                    CutReason(
                        kind="fallback_window",
                        confidence=0.2,
                        rationale=f"target_scene_seconds={target_scene_seconds}",
                    )
                ],
            )
        )
        timestamp += target_scene_seconds
    return cuts


def _source_lifecycle_cuts(
    *,
    probe: VideoProbe,
    tracks: list[Sam3EntityTrack],
) -> list[CandidateCut]:
    cuts: list[CandidateCut] = []
    for track in tracks:
        for offset, timestamp in enumerate((track.start_seconds, track.end_seconds)):
            if timestamp <= 0.0 or timestamp >= probe.duration_seconds:
                continue
            confidence = round(max(0.35, track.confidence), 3)
            cuts.append(
                CandidateCut(
                    cut_id=f"source-lifecycle-{track.track_id}-{offset}",
                    timestamp_seconds=round(timestamp, 3),
                    confidence=confidence,
                    reasons=[
                        CutReason(
                            kind="source_lifecycle_change",
                            confidence=confidence,
                            rationale=f"{track.track_id} {'appears' if offset == 0 else 'disappears'}",
                        )
                    ],
                )
            )
    return cuts


def _label_context_cuts(
    *,
    frame_batches: list[FrameBatch],
    tracks: list[Sam3EntityTrack],
    label_candidates_by_track: dict[str, list[LabelCandidate]],
) -> list[CandidateCut]:
    cuts: list[CandidateCut] = []
    track_ids_by_scene: dict[int, list[str]] = {}
    for track in tracks:
        track_ids_by_scene.setdefault(track.scene_index, []).append(track.track_id)
    scene_labels: list[tuple[int, str | None, float]] = []
    for batch in frame_batches:
        labels: list[LabelCandidate] = []
        for track_id in track_ids_by_scene.get(batch.scene_index, []):
            labels.extend(label_candidates_by_track.get(track_id, [])[:1])
        labels.sort(key=lambda candidate: candidate.score, reverse=True)
        scene_labels.append(
            (
                batch.scene_index,
                labels[0].label if labels else None,
                labels[0].score if labels else 0.0,
            )
        )
    for left, right in zip(scene_labels, scene_labels[1:], strict=False):
        if left[1] is None or right[1] is None or left[1] == right[1]:
            continue
        right_batch = next(
            (batch for batch in frame_batches if batch.scene_index == right[0]),
            None,
        )
        if right_batch is None or not right_batch.frames:
            continue
        confidence = round(min(1.0, max(left[2], right[2])), 3)
        cuts.append(
            CandidateCut(
                cut_id=f"label-context-{left[0]}-{right[0]}",
                timestamp_seconds=round(right_batch.frames[0].timestamp_seconds, 3),
                confidence=confidence,
                reasons=[
                    CutReason(
                        kind="label_context_change",
                        confidence=confidence,
                        rationale=f"{left[1]} -> {right[1]}",
                    )
                ],
            )
        )
    return cuts


def _interaction_onset_cuts(
    *,
    probe: VideoProbe,
    tracks: list[Sam3EntityTrack],
) -> list[CandidateCut]:
    cuts: list[CandidateCut] = []
    for track in tracks:
        if track.features.interaction_score < 0.6:
            continue
        if track.start_seconds <= 0.0 or track.start_seconds >= probe.duration_seconds:
            continue
        confidence = round(
            min(1.0, max(track.features.interaction_score, track.confidence)),
            3,
        )
        cuts.append(
            CandidateCut(
                cut_id=f"interaction-onset-{track.track_id}",
                timestamp_seconds=round(track.start_seconds, 3),
                confidence=confidence,
                reasons=[
                    CutReason(
                        kind="interaction_onset",
                        confidence=confidence,
                        rationale=f"{track.track_id} interaction_score={track.features.interaction_score:.2f}",
                    )
                ],
            )
        )
    return cuts


def _merge_cut_group(cut_group: list[CandidateCut], group_index: int) -> CandidateCut:
    total_confidence = sum(max(cut.confidence, 0.001) for cut in cut_group)
    weighted_timestamp = sum(
        cut.timestamp_seconds * max(cut.confidence, 0.001)
        for cut in cut_group
    ) / total_confidence
    reasons: list[CutReason] = []
    seen_reasons: set[tuple[str, str]] = set()
    for cut in cut_group:
        for reason in cut.reasons:
            key = (reason.kind, reason.rationale)
            if key in seen_reasons:
                continue
            seen_reasons.add(key)
            reasons.append(reason)
    confidence = max(cut.confidence for cut in cut_group)
    return CandidateCut(
        cut_id=f"cut-{group_index:04d}",
        timestamp_seconds=round(weighted_timestamp, 3),
        confidence=round(confidence, 3),
        reasons=reasons,
    )


def _scene_sample_timestamps(
    scene: SceneBoundary,
    *,
    frames_per_scene: int,
) -> list[float]:
    if frames_per_scene <= 0:
        return []
    duration = max(scene.end_seconds - scene.start_seconds, 0.0)
    if duration == 0.0:
        return [scene.start_seconds]

    step = duration / (frames_per_scene + 1)
    return [
        round(scene.start_seconds + step * index, 3)
        for index in range(1, frames_per_scene + 1)
    ]


def _parse_frame_rate(raw_rate: str | None) -> float | None:
    if not raw_rate or raw_rate == "0/0":
        return None
    if "/" not in raw_rate:
        return float(raw_rate)
    numerator, denominator = raw_rate.split("/", 1)
    if denominator == "0":
        return None
    return float(numerator) / float(denominator)
