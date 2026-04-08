# Stage 2: Candidate Cuts, Evidence Windows, and Storyboard Artifacts

**Suggested duration:** 4-6 focused days

## Goal

Replace the current fixed-window scene planning with a structural layer that proposes candidate cuts and builds evidence windows the rest of the system can trust. The output of this stage is not final semantic segmentation; it is a high-recall structural scaffold.

## Why this stage exists

The current fixed 5-second windows are useful for bootstrapping, but they are not a meaningful answer to “where should the pipeline cut?” They force downstream logic to inherit arbitrary boundaries. The final system needs explicit candidate cuts with reasons and confidence so the agent can decide whether to keep, merge, or split them.

## Inputs

- A stable repository from Stage 1.
- The Stage 0 schema definitions for candidate cuts and evidence windows.
- The current `probe_video`, `detect_scenes`, and `sample_frames` utilities as starting points.

## Expected outputs

- Shot-boundary proposals.
- Candidate cuts with typed reasons and confidence.
- Evidence windows with sampled frames and optional short visual clips.
- A storyboard artifact for each video.

## Primary code areas

- `src/v2a_inspect/tools/media.py` or a new server-side cuts module.
- New contract types for `CandidateCut` and `EvidenceWindow`.
- New tests and fixture outputs for boundary generation.

## Explicitly out of scope

- No final source tracking yet.
- No cross-scene identity logic yet.
- No final grouping or routing.

## Main workstreams

### 1. Ship hard cuts first, then soft cuts

Do not block the stage on perfect semantics. First implement reliable hard cut proposals from shot-boundary tools such as ffmpeg or PySceneDetect-style histogram differences. Once that works, add soft candidate cuts based on motion changes, scene-composition changes, or other cheap structural cues. Source-lifecycle cuts that depend on track extraction can be layered in later stages without invalidating the basic contract.

### 2. Create evidence windows as first-class artifacts

An evidence window is not just a time range. It must include sampled frame IDs, optional short clip paths, the cut reasons that created it, and enough metadata for the agent or reviewer to inspect it later. Every later stage should refer back to evidence windows instead of treating timing as anonymous floats.

### 3. Build a storyboard artifact

For each video, export a compact storyboard that shows representative frames per evidence window in temporal order. This gives the human team and the agent a cheap global overview and reduces the temptation to send whole videos to Gemini for every decision.

### 4. Merge and clean the cut graph

Candidate cuts will often cluster tightly. Add deterministic merge rules for duplicate or near-duplicate proposals, minimum-window-length rules, and fallback behavior for extremely short clips. The goal is high recall without exploding the number of windows.


## Exit criteria

- Fixed 5-second windows are no longer the only structural unit.
- The system can export a human-readable storyboard and cut list.
- Evidence windows are tied to explicit reasons for existence.
- The planner can consume the cut list without reading the whole video first.

## How to use this stage folder

1. Read [the detailed stage plan](stage_2_detailed.md).
2. Use [the stage checklist](stage_2_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means the pipeline has a structural scaffold that is interpretable and inspectable. The rest of the system can now build on evidence windows rather than arbitrary time slices.
