# Stage 2 Detailed Plan: Candidate Cuts, Evidence Windows, and Storyboard Artifacts

[Back to stage main](stage_2_main.md) · [Go to checklist](stage_2_checklist.md)

## Intended result

By the end of Stage 2, the project should have crossed a clear boundary: Done means the pipeline has a structural scaffold that is interpretable and inspectable. The rest of the system can now build on evidence windows rather than arbitrary time slices.

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- A stable repository from Stage 1.
- The Stage 0 schema definitions for candidate cuts and evidence windows.
- The current `probe_video`, `detect_scenes`, and `sample_frames` utilities as starting points.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- Shot-boundary proposals.
- Candidate cuts with typed reasons and confidence.
- Evidence windows with sampled frames and optional short visual clips.
- A storyboard artifact for each video.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Ship hard cuts first, then soft cuts

Do not block the stage on perfect semantics. First implement reliable hard cut proposals from shot-boundary tools such as ffmpeg or PySceneDetect-style histogram differences. Once that works, add soft candidate cuts based on motion changes, scene-composition changes, or other cheap structural cues. Source-lifecycle cuts that depend on track extraction can be layered in later stages without invalidating the basic contract.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Create evidence windows as first-class artifacts

An evidence window is not just a time range. It must include sampled frame IDs, optional short clip paths, the cut reasons that created it, and enough metadata for the agent or reviewer to inspect it later. Every later stage should refer back to evidence windows instead of treating timing as anonymous floats.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Build a storyboard artifact

For each video, export a compact storyboard that shows representative frames per evidence window in temporal order. This gives the human team and the agent a cheap global overview and reduces the temptation to send whole videos to Gemini for every decision.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Merge and clean the cut graph

Candidate cuts will often cluster tightly. Add deterministic merge rules for duplicate or near-duplicate proposals, minimum-window-length rules, and fallback behavior for extremely short clips. The goal is high recall without exploding the number of windows.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Add contract models such as `CandidateCut`, `CutReason`, and `EvidenceWindow`.
- Replace or wrap `detect_scenes()` with a richer candidate-cut builder.
- Add helper functions to create storyboard images and short clips.
- Update the interim server context builder to emit the new structural artifacts.

## Implementation notes for the current repository

- The current `detect_scenes()` fixed-window implementation should become either a fallback or an adapter.
- Introduce typed cut reasons such as `shot_boundary`, `composition_change`, `motion_regime_change`, and `fallback_window`.
- Keep the initial algorithm simple. A high-recall structural scaffold is enough for this stage.
- Export storyboard artifacts to a predictable location so later agent and UI code can reference them.
- Do not pretend candidate cuts are final event boundaries; keep confidence and provenance on every proposed cut.
- Merge aggressively only at the structural level; semantic merge/split decisions should remain later responsibilities.

## What must not happen in this stage

- No final source tracking yet.
- No cross-scene identity logic yet.
- No final grouping or routing.

In addition to the items above, do not silently introduce new semantics that belong to later stages.

## Suggested milestone breakdown inside the stage

A useful internal cadence for this stage is:

1. add or modify contracts and tests first
2. implement the minimal code path
3. run fixtures and inspect artifacts manually
4. only then refine thresholds or prompts
5. update docs and adapters before declaring the stage complete

## Tests and measurement for this stage

A stage is only meaningful if there is a way to tell whether it worked.

- Deterministic tests for boundary generation on fixture clips.
- Minimum/maximum window-length tests.
- Storyboard generation tests.
- Merge-rule tests for near-duplicate cuts.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- The system proposes too many cuts and becomes expensive.
- The system proposes too few cuts and misses important source changes.
- Developers mistake candidate cuts for final event boundaries.

## Questions the agent should ask before merging changes

- Are candidate cuts explicitly labeled as proposals rather than final truth?
- Can a human inspect the storyboard and understand why windows exist?
- Did we remove the silent dependence on arbitrary fixed windows?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means the pipeline has a structural scaffold that is interpretable and inspectable. The rest of the system can now build on evidence windows rather than arbitrary time slices.
