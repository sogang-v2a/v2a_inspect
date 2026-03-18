"""
track_grouper.py — Cross-scene track grouping for temporal consistency.

Pipeline:
  1. extract_raw_tracks()       — flatten VideoSceneAnalysis → list[RawTrack]
  2. group_tracks_by_text()     — single Gemini call groups semantically similar tracks
  3. verify_groups_with_vlm()   — optional: Gemini VLM confirms groups visually
  4. assign_model_selections()  — optional: Gemini VLM assigns TTA/VTA per raw track
  5. build_grouped_analysis()   — inject group_id / canonical_description into scene analysis
  6. group_tracks()             — top-level entry combining steps 1-5
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import time as _time

from google.genai import types

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt from the prompts/ directory."""
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Retry helper with timeout
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_MS = 120_000   # 2 min – text-only calls
_VLM_TIMEOUT_MS = 180_000      # 3 min – video + text calls
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0        # seconds


def _generate_with_retry(
    client,
    model: str,
    contents: list,
    config: types.GenerateContentConfig | None = None,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    max_retries: int = _MAX_RETRIES,
    label: str = "",
):
    """Wrapper around generate_content with HTTP timeout and exponential backoff."""
    if config is None:
        config = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
    else:
        config.http_options = types.HttpOptions(timeout=timeout_ms)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config,
            )
        except Exception as e:
            last_exc = e
            err = str(e).lower()
            transient = any(
                k in err for k in ("429", "503", "timeout", "deadline", "unavailable")
            )
            if transient and attempt < max_retries:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[track_grouper] {label} attempt {attempt} failed: {e}. "
                    f"Retry in {delay:.0f}s..."
                )
                _time.sleep(delay)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelSelection:
    """TTA/VTA model selection result for a single track or group."""
    model_type: str          # "TTA" | "VTA"
    confidence: float        # 0.0–1.0
    vta_score: float         # combined VTA preference (video motion + event coupling)
    tta_score: float         # combined TTA preference (source diversity + object count bias)
    reasoning: str
    rule_based: bool = False # True = deterministic rule (background, etc.), False = LLM judgment


@dataclass
class RawTrack:
    """One track extracted from a Scene (background or object)."""
    track_id: str          # e.g. "s0_bg", "s0_obj0", "s1_obj1"
    scene_index: int
    kind: str              # "background" | "object"
    description: str
    start: float
    end: float
    obj_index: Optional[int] = None       # None for backgrounds
    n_scene_objects: int = 0              # number of object tracks in the same scene
    model_selection: Optional[ModelSelection] = None  # assigned post-grouping

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TrackGroup:
    """A set of RawTracks that represent the same real-world audio entity."""
    group_id: str
    canonical_description: str     # description used for audio generation
    member_ids: list[str]          # track_ids belonging to this group
    vlm_verified: bool = False
    model_selection: Optional[ModelSelection] = None  # group-level representative


@dataclass
class GroupedAnalysis:
    """VideoSceneAnalysis annotated with group assignments."""
    scene_analysis: object          # VideoSceneAnalysis (annotated copy)
    raw_tracks: list[RawTrack]
    groups: list[TrackGroup]
    track_to_group: dict[str, str]  # track_id -> group_id


# ---------------------------------------------------------------------------
# Step 1: Extract flat track list from VideoSceneAnalysis
# ---------------------------------------------------------------------------

def extract_raw_tracks(scene_analysis) -> list[RawTrack]:
    """Flatten VideoSceneAnalysis into an ordered list of RawTrack.

    Order per scene: background first, then objects in order.
    """
    tracks: list[RawTrack] = []
    for scene in scene_analysis.scenes:
        si = scene.scene_index
        n_objs = len(scene.objects)
        tracks.append(RawTrack(
            track_id=f"s{si}_bg",
            scene_index=si,
            kind="background",
            description=scene.background_sound,
            start=scene.time_range.start,
            end=scene.time_range.end,
            obj_index=None,
            n_scene_objects=n_objs,
        ))
        for oi, obj in enumerate(scene.objects):
            tracks.append(RawTrack(
                track_id=f"s{si}_obj{oi}",
                scene_index=si,
                kind="object",
                description=obj.description,
                start=obj.time_range.start,
                end=obj.time_range.end,
                obj_index=oi,
                n_scene_objects=n_objs,
            ))
    return tracks


# ---------------------------------------------------------------------------
# Step 2: Text-based semantic grouping via Gemini
# ---------------------------------------------------------------------------

_GROUPING_PROMPT_TEMPLATE = _load_prompt("grouping.txt")

def _build_numbered_list(tracks: list[RawTrack]) -> str:
    lines = []
    for i, t in enumerate(tracks):
        lines.append(f"[{i}] {t.track_id} ({t.kind}, scene {t.scene_index}, {t.start:.1f}s-{t.end:.1f}s): {t.description}")
    return "\n".join(lines)


def _strip_markdown_json(text: str) -> str:
    """Strip markdown code block wrapper if present (```json ... ``` or ``` ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[: text.rfind("```")]
    return text.strip()


def _parse_grouping_response(response_text: str, num_tracks: int) -> list[list[int]]:
    """Parse Gemini grouping response. Returns list of index groups.

    Falls back to singletons for any missing/invalid indices.
    Handles markdown code block wrappers (```json ... ```).
    """
    try:
        data = json.loads(_strip_markdown_json(response_text))
        raw_groups = data.get("groups", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"[track_grouper] Failed to parse grouping response, using singletons. Response: {response_text[:200]}")
        return [[i] for i in range(num_tracks)]

    # Collect all indices that appear in groups
    seen: set[int] = set()
    parsed: list[list[int]] = []
    for g in raw_groups:
        members = g.get("member_indices", [])
        valid = [m for m in members if isinstance(m, int) and 0 <= m < num_tracks and m not in seen]
        if valid:
            parsed.append(valid)
            seen.update(valid)

    # Singleton fallback for any missing indices
    for i in range(num_tracks):
        if i not in seen:
            parsed.append([i])

    return parsed


def _extract_canonical_indices(response_text: str, groups_by_members: list[list[int]]) -> dict[int, int]:
    """Return {group_position: canonical_index} from response.

    Falls back to first member if parsing fails.
    """
    canonical_map: dict[int, int] = {}
    try:
        data = json.loads(_strip_markdown_json(response_text))
        raw_groups = data.get("groups", [])
        # Build a lookup: frozenset(members) -> canonical_index
        member_to_canonical: dict[frozenset, int] = {}
        for g in raw_groups:
            members = tuple(g.get("member_indices", []))
            ci = g.get("canonical_index")
            if members and isinstance(ci, int):
                member_to_canonical[frozenset(members)] = ci
    except Exception:
        return {}

    for pos, group in enumerate(groups_by_members):
        fs = frozenset(group)
        ci = member_to_canonical.get(fs)
        if ci is not None and ci in group:
            canonical_map[pos] = ci
        else:
            canonical_map[pos] = group[0]

    return canonical_map


def group_tracks_by_text(
    raw_tracks: list[RawTrack],
    client,
    model: str = "gemini-3-pro-preview",
) -> list[TrackGroup]:
    """Group tracks semantically using a single Gemini text call.

    Returns a list of TrackGroup with canonical_description set.
    vlm_verified is False for all groups at this stage.
    """
    if not raw_tracks:
        return []

    numbered_list = _build_numbered_list(raw_tracks)
    prompt = _GROUPING_PROMPT_TEMPLATE.format(numbered_list=numbered_list)

    print(f"[track_grouper] Requesting semantic grouping for {len(raw_tracks)} tracks...")
    try:
        response = _generate_with_retry(
            client, model=model, contents=[prompt],
            timeout_ms=_DEFAULT_TIMEOUT_MS, label="text_grouping",
        )
        response_text = response.text
    except Exception as e:
        print(f"[track_grouper] Gemini grouping call failed ({type(e).__name__}): {e}. Falling back to singletons.")
        response_text = '{"groups": []}'

    index_groups = _parse_grouping_response(response_text, len(raw_tracks))
    canonical_map = _extract_canonical_indices(response_text, index_groups)

    groups: list[TrackGroup] = []
    for pos, member_indices in enumerate(index_groups):
        canonical_idx = canonical_map.get(pos, member_indices[0])
        canonical_desc = raw_tracks[canonical_idx].description
        member_ids = [raw_tracks[i].track_id for i in member_indices]
        groups.append(TrackGroup(
            group_id=f"g{pos}",
            canonical_description=canonical_desc,
            member_ids=member_ids,
            vlm_verified=False,
        ))

    print(f"[track_grouper] Text grouping: {len(raw_tracks)} tracks → {len(groups)} groups")
    for g in groups:
        if len(g.member_ids) > 1:
            print(f"  {g.group_id}: {g.member_ids} → '{g.canonical_description[:60]}...'")

    return groups


# ---------------------------------------------------------------------------
# Step 3: VLM visual verification via Gemini
# ---------------------------------------------------------------------------

_VLM_VERIFY_PROMPT_TEMPLATE = _load_prompt("vlm_verify.txt")

def _build_segment_list(group: TrackGroup, tracks_by_id: dict[str, RawTrack]) -> str:
    lines = []
    for i, tid in enumerate(group.member_ids):
        t = tracks_by_id[tid]
        lines.append(f"  Segment {i}: scene {t.scene_index}, {t.start:.1f}s-{t.end:.1f}s | \"{t.description}\"")
    return "\n".join(lines)


def verify_groups_with_vlm(
    groups: list[TrackGroup],
    raw_tracks: list[RawTrack],
    video_file,
    client,
    model: str = "gemini-3-pro-preview",
    progress_callback=None,
) -> list[TrackGroup]:
    """Visually verify multi-member groups using Gemini VLM.

    - Skips singleton groups (nothing to verify).
    - Skips same-scene-only groups (no cross-scene comparison needed).
    - Splits groups if VLM disagrees; assigns new group IDs like "g2_a", "g2_b".
    - Returns updated groups list (new list, original untouched).
    """
    if video_file is None:
        print("[track_grouper] VLM verify skipped: no video_file provided.")
        return groups

    tracks_by_id = {t.track_id: t for t in raw_tracks}

    # Build Gemini video part (reuse the already-uploaded file)
    video_part = types.Part(
        file_data=types.FileData(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
        ),
        video_metadata=types.VideoMetadata(fps=2.0),
    )

    vlm_targets = [g for g in groups if len(g.member_ids) >= 2]
    updated_groups: list[TrackGroup] = []
    vlm_idx = 0
    for g in groups:
        # Skip singletons
        if len(g.member_ids) < 2:
            g.vlm_verified = False
            updated_groups.append(g)
            continue

        # Only verify cross-scene groups (members from >=2 different scenes)
        member_scenes = {tracks_by_id[tid].scene_index for tid in g.member_ids if tid in tracks_by_id}
        if len(member_scenes) < 2:
            g.vlm_verified = False
            updated_groups.append(g)
            continue

        segment_list = _build_segment_list(g, tracks_by_id)
        prompt = _VLM_VERIFY_PROMPT_TEMPLATE.format(
            canonical_description=g.canonical_description,
            segment_list=segment_list,
        )

        vlm_idx += 1
        if progress_callback:
            progress_callback(f"  VLM 검증: 그룹 {vlm_idx}/{len(vlm_targets)} ({g.group_id})...")
        print(f"[track_grouper] VLM verifying group {g.group_id} ({len(g.member_ids)} members)...")
        try:
            response = _generate_with_retry(
                client, model=model, contents=[video_part, prompt],
                timeout_ms=_VLM_TIMEOUT_MS, label=f"vlm_verify_{g.group_id}",
            )
            data = json.loads(_strip_markdown_json(response.text))
            same_entity = data.get("same_entity", True)
            reasoning = data.get("reasoning", "")
            print(f"  → same_entity={same_entity}. Reason: {reasoning}")
        except Exception as e:
            print(f"  → VLM call failed ({type(e).__name__}): {e}. Keeping group as-is.")
            g.vlm_verified = False
            updated_groups.append(g)
            continue

        if same_entity is True or same_entity == "uncertain":
            # Keep group, mark as verified (or uncertain = best-effort)
            g.vlm_verified = (same_entity is True)
            updated_groups.append(g)
        else:
            # Split the group according to confirmed_groups
            confirmed = data.get("confirmed_groups")
            if not confirmed or not isinstance(confirmed, list):
                # Fallback: singletons
                confirmed = [[i] for i in range(len(g.member_ids))]

            for sub_idx, sub_members in enumerate(confirmed):
                valid_indices = [i for i in sub_members if isinstance(i, int) and 0 <= i < len(g.member_ids)]
                if not valid_indices:
                    continue
                sub_track_ids = [g.member_ids[i] for i in valid_indices]
                # Pick best canonical from sub-group
                sub_tracks = [tracks_by_id[tid] for tid in sub_track_ids if tid in tracks_by_id]
                canon_desc = max(sub_tracks, key=lambda t: len(t.description)).description if sub_tracks else g.canonical_description
                updated_groups.append(TrackGroup(
                    group_id=f"{g.group_id}_{chr(ord('a') + sub_idx)}",
                    canonical_description=canon_desc,
                    member_ids=sub_track_ids,
                    vlm_verified=True,
                ))

            # Any member_ids not covered by confirmed_groups → own singleton group
            covered = {g.member_ids[i] for sub in confirmed for i in sub if isinstance(i, int) and 0 <= i < len(g.member_ids)}
            uncovered = [tid for tid in g.member_ids if tid not in covered]
            for sub_idx, tid in enumerate(uncovered):
                t = tracks_by_id.get(tid)
                updated_groups.append(TrackGroup(
                    group_id=f"{g.group_id}_uc{sub_idx}",
                    canonical_description=t.description if t else g.canonical_description,
                    member_ids=[tid],
                    vlm_verified=True,
                ))

    print(f"[track_grouper] After VLM verify: {len(groups)} → {len(updated_groups)} groups")
    return updated_groups


# ---------------------------------------------------------------------------
# Step 4: Inject group metadata into scene analysis
# ---------------------------------------------------------------------------

def build_grouped_analysis(
    scene_analysis,
    raw_tracks: list[RawTrack],
    groups: list[TrackGroup],
) -> GroupedAnalysis:
    """Annotate a deep copy of scene_analysis with group_id and canonical_description fields.

    Returns GroupedAnalysis with the annotated scene_analysis and lookup tables.
    """
    # Build reverse lookup: track_id -> TrackGroup
    track_to_group: dict[str, str] = {}
    track_to_canon: dict[str, str] = {}
    for g in groups:
        for tid in g.member_ids:
            track_to_group[tid] = g.group_id
            track_to_canon[tid] = g.canonical_description

    # Deep copy scene_analysis to avoid mutating the original
    annotated = copy.deepcopy(scene_analysis)

    for scene in annotated.scenes:
        si = scene.scene_index
        bg_tid = f"s{si}_bg"
        scene.background_group_id = track_to_group.get(bg_tid)
        scene.background_canonical = track_to_canon.get(bg_tid)

        for oi, obj in enumerate(scene.objects):
            obj_tid = f"s{si}_obj{oi}"
            obj.group_id = track_to_group.get(obj_tid)
            obj.canonical_description = track_to_canon.get(obj_tid)

    return GroupedAnalysis(
        scene_analysis=annotated,
        raw_tracks=raw_tracks,
        groups=groups,
        track_to_group=track_to_group,
    )


# ---------------------------------------------------------------------------
# Step 4: TTA/VTA model selection via Gemini VLM
# ---------------------------------------------------------------------------

_MODEL_SELECT_PROMPT_TEMPLATE = _load_prompt("model_select.txt")


def _select_model_from_scores(
    motion: float,
    coupling: float,
    source_div: float,
    n_objects: int = 0,
    duration: float = 0.0,
) -> tuple[str, float, float, float]:
    """Determine TTA or VTA from video-observable scores plus deterministic signals.

    VTA preference driven by: visual dynamism (motion) + tight event-sound coupling.
    TTA preference driven by: source separation need (source_diversity + object count).

    Additional deterministic corrections:
      - n_objects >= 2: +1.0 to tta_raw (more objects → TTA preferred)
      - n_objects >= 3: +0.5 further
      - duration < 1.0s: +0.5 to vta_raw (short impact clip → sync matters more)

    Returns:
        (model_type, confidence, vta_combined, tta_combined)
    """
    vta_raw = (motion + coupling) / 2.0

    tta_raw = source_div
    if n_objects >= 2:
        tta_raw += 1.0
    if n_objects >= 3:
        tta_raw += 0.5

    if duration > 0.0 and duration < 1.0:
        vta_raw += 0.5

    diff = vta_raw - tta_raw
    if diff >= 1.5:
        confidence = min(0.5 + diff * 0.15, 0.95)
        return "VTA", round(confidence, 3), round(vta_raw, 2), round(tta_raw, 2)
    elif diff <= -1.5:
        confidence = min(0.5 + (-diff) * 0.15, 0.95)
        return "TTA", round(confidence, 3), round(vta_raw, 2), round(tta_raw, 2)
    else:
        return "TTA", 0.5, round(vta_raw, 2), round(tta_raw, 2)  # default: TTA (quality)


def assign_model_selections(
    groups: list[TrackGroup],
    raw_tracks: list[RawTrack],
    video_file,
    client,
    model: str = "gemini-3-pro-preview",
    progress_callback=None,
) -> tuple[list[TrackGroup], list[RawTrack]]:
    """Assign TTA/VTA model to each raw track and each group using Gemini VLM.

    For each group, all member segments are evaluated together so the model
    has cross-scene context. Group-level model is derived from the average
    sync/isolation scores across members. Per-member ModelSelection is stored
    on each RawTrack; group representative is stored on TrackGroup.

    Groups with conflicting member models (confidence < 0.6) are flagged via
    lower group confidence.

    Returns:
        (updated_groups, updated_raw_tracks) — new lists with model_selection populated.
    """
    if video_file is None:
        print("[track_grouper] Model selection skipped: no video_file provided.")
        return groups, raw_tracks

    tracks_by_id: dict[str, RawTrack] = {t.track_id: t for t in raw_tracks}

    video_part = types.Part(
        file_data=types.FileData(
            file_uri=video_file.uri,
            mime_type=video_file.mime_type,
        ),
        video_metadata=types.VideoMetadata(fps=2.0),
    )

    # Process group-by-group
    for gi, g in enumerate(groups):
        member_tracks = [tracks_by_id[tid] for tid in g.member_ids if tid in tracks_by_id]
        if not member_tracks:
            continue

        # ── Rule-based: background groups → TTA directly, skip LLM ──────────
        if all(t.kind == "background" for t in member_tracks):
            bg_reasoning = (
                "Background track: TTA preferred to avoid foreground object sound bleed-through "
                "that VTA may introduce by attending to visible objects."
            )
            for t in member_tracks:
                t.model_selection = ModelSelection(
                    model_type="TTA", confidence=0.90,
                    vta_score=1.0, tta_score=5.0,
                    reasoning=bg_reasoning,
                    rule_based=True,
                )
            avg_n_objs = sum(t.n_scene_objects for t in member_tracks) / len(member_tracks)
            g.model_selection = ModelSelection(
                model_type="TTA", confidence=0.90,
                vta_score=1.0, tta_score=5.0,
                reasoning=bg_reasoning,
                rule_based=True,
            )
            print(f"[track_grouper] Group {g.group_id}: background → TTA (rule-based, skipping LLM)")
            continue

        # ── LLM path: object tracks ──────────────────────────────────────────
        seg_lines = [
            f"  Segment {i}: scene {t.scene_index}, {t.start:.1f}s-{t.end:.1f}s"
            f" | kind={t.kind} | n_objects_in_scene={t.n_scene_objects}"
            f" | \"{t.description}\""
            for i, t in enumerate(member_tracks)
        ]
        segment_list = "\n".join(seg_lines)
        prompt = _MODEL_SELECT_PROMPT_TEMPLATE.format(segment_list=segment_list)

        if progress_callback:
            progress_callback(f"  모델 선정: 그룹 {gi + 1}/{len(groups)} ({g.group_id})...")
        print(f"[track_grouper] Model select for group {g.group_id} ({len(member_tracks)} segments)...")
        try:
            response = _generate_with_retry(
                client, model=model, contents=[video_part, prompt],
                timeout_ms=_VLM_TIMEOUT_MS, label=f"model_select_{g.group_id}",
            )
            data = json.loads(_strip_markdown_json(response.text))
            seg_results = data.get("segments", [])
        except Exception as e:
            print(f"  → Model select call failed ({type(e).__name__}): {e}. Skipping group.")
            continue

        # Map segment_index → member_track and build per-track ModelSelection
        vta_scores: list[float] = []
        tta_scores: list[float] = []

        for seg in seg_results:
            idx = seg.get("segment_index")
            if idx is None or idx >= len(member_tracks):
                continue
            track = member_tracks[idx]
            motion = float(seg.get("motion_level", 3))
            coupling = float(seg.get("event_coupling", 3))
            src_div = float(seg.get("source_diversity", 3))
            reasoning = seg.get("reasoning", "")
            model_type, conf, vta_val, tta_val = _select_model_from_scores(
                motion, coupling, src_div,
                n_objects=track.n_scene_objects,
                duration=track.duration,
            )
            track.model_selection = ModelSelection(
                model_type=model_type,
                confidence=conf,
                vta_score=vta_val,
                tta_score=tta_val,
                reasoning=reasoning,
                rule_based=False,
            )
            vta_scores.append(vta_val)
            tta_scores.append(tta_val)
            print(
                f"  Segment {idx} ({track.track_id}): "
                f"motion={motion}, coupling={coupling}, src_div={src_div} "
                f"→ vta={vta_val:.1f}/tta={tta_val:.1f} → {model_type} (conf={conf:.0%})"
            )

        # Group-level: average vta/tta scores of evaluated members
        if vta_scores:
            avg_vta = sum(vta_scores) / len(vta_scores)
            avg_tta = sum(tta_scores) / len(tta_scores)
            diff = avg_vta - avg_tta
            if diff >= 1.5:
                group_model = "VTA"
                group_conf = min(0.5 + diff * 0.15, 0.95)
            elif diff <= -1.5:
                group_model = "TTA"
                group_conf = min(0.5 + (-diff) * 0.15, 0.95)
            else:
                group_model, group_conf = "TTA", 0.5

            # Detect within-group disagreement
            member_models = {t.model_selection.model_type for t in member_tracks if t.model_selection}
            if len(member_models) > 1:
                group_conf = min(group_conf, 0.55)

            all_reasoning = "; ".join(
                t.model_selection.reasoning
                for t in member_tracks
                if t.model_selection and t.model_selection.reasoning
            )
            g.model_selection = ModelSelection(
                model_type=group_model,
                confidence=round(group_conf, 3),
                vta_score=round(avg_vta, 2),
                tta_score=round(avg_tta, 2),
                reasoning=all_reasoning[:200],
                rule_based=False,
            )
            print(f"  → Group {g.group_id}: {group_model} (vta={avg_vta:.1f}, tta={avg_tta:.1f}, conf={group_conf:.0%})")

    # Rebuild raw_tracks list preserving order (model_selection mutated in-place above)
    return groups, raw_tracks


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def group_tracks(
    scene_analysis,
    client,
    video_file=None,
    enable_vlm_verify: bool = True,
    enable_model_select: bool = False,
    model: str = "gemini-3-pro-preview",
    progress_callback=None,
) -> GroupedAnalysis:
    """Full grouping pipeline: extract → text-group → [vlm-verify] → [model-select] → assemble.

    Args:
        scene_analysis: VideoSceneAnalysis object from analyze_video()
        client: Gemini genai.Client
        video_file: Gemini uploaded file object (required for VLM verify / model select)
        enable_vlm_verify: If True and video_file is provided, run VLM verification
        enable_model_select: If True and video_file is provided, assign TTA/VTA per track
        model: Gemini model name

    Returns:
        GroupedAnalysis with annotated scene_analysis and group metadata
    """
    raw_tracks = extract_raw_tracks(scene_analysis)

    if not raw_tracks:
        return GroupedAnalysis(
            scene_analysis=copy.deepcopy(scene_analysis),
            raw_tracks=[],
            groups=[],
            track_to_group={},
        )

    groups = group_tracks_by_text(raw_tracks, client, model=model)

    if enable_vlm_verify and video_file is not None:
        if progress_callback:
            progress_callback("VLM 검증 시작...")
        groups = verify_groups_with_vlm(
            groups, raw_tracks, video_file, client, model=model,
            progress_callback=progress_callback,
        )

    if enable_model_select and video_file is not None:
        if progress_callback:
            progress_callback("모델 선정 시작...")
        groups, raw_tracks = assign_model_selections(
            groups, raw_tracks, video_file, client, model=model,
            progress_callback=progress_callback,
        )

    return build_grouped_analysis(scene_analysis, raw_tracks, groups)
