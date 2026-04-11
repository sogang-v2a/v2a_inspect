# Stage 3 Detailed Plan: Crop-Based Source Extraction, Re-Identification, and Identity Confidence

[Back to stage main](stage_3_main.md) · [Go to checklist](stage_3_checklist.md)

## Intended result

By the end of Stage 3, the project should have crossed a clear boundary: Done means there is a trustworthy visual identity layer. The system can point to actual crops and say, “this is the visual evidence behind source X.”

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- Evidence windows from Stage 2.
- SAM3, DINOv2, and SigLIP2 runtimes already introduced on the server side.
- The contract rule that manual text-conditioned extraction is recovery-only at the external interface.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- Track crops generated from masks or boxes.
- Stable source-track IDs and identity confidence.
- DINOv2 embeddings computed on crops.
- SigLIP2 label scores computed on crops.
- A re-identification graph across windows and cuts.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Make crop generation real

Implement `crop_tracks` as a first-class tool. Crops should be created from masks when possible and from boxes with padding when masks are missing. Store both the original frame reference and the crop artifact. Later stages must be able to inspect which visual evidence supported a source track.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Define the external scene-prompt contract

The blueprint now treats default extraction as scene-prompt-narrowed, using SigLIP2 label suggestions or another model-scored visual vocabulary before SAM3 runs. Manual text recovery must remain a separate tool path.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Build re-identification as a graph, not a one-shot cluster

Use DINOv2 crop embeddings, temporal adjacency, label compatibility, and window continuity to create an identity graph. Same-window duplicates should be hard to merge; cross-window and cross-cut edges should require stronger evidence. Keep confidence on every identity edge so the agent can later inspect ambiguous merges.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Expose ambiguity

Not every visible source will be easy to track. When identity is weak, store that weakness explicitly rather than silently forcing a merge. A low-confidence source split is often safer than an unjustified merge, especially before generation grouping.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Create a crop-generation module, e.g. `server/tools/crop_tracks.py`.
- Update the SAM3 extraction client to store enough geometry for crop building.
- Create a re-id module, e.g. `server/tools/reid.py`, that returns identity edges or provisional source tracks.
- Update label and embedding modules to operate on crop paths.
- Keep the old `tool_context` path as an adapter only until Stage 5.

## Implementation notes for the current repository

- Implement `crop_tracks` before trying to improve grouping thresholds. Better evidence beats threshold-tuning.
- If the SAM runtime requires prompts internally, wrap that behind an external default-extraction tool that does not require manual prompting.
- Use crop provenance: every crop should know which source track, window, frame, and geometry produced it.
- Build re-identification as a graph or edge set with confidence rather than one irreversible clustering pass.
- Keep recovery prompts separate and explicitly logged.
- When evidence is weak, prefer preserving ambiguity over forcing a brittle identity merge.

## What must not happen in this stage

- No final event segmentation yet.
- No final generation groups yet.
- No full agentic tool loop yet.

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

- Crop-generation tests with synthetic boxes and real fixture frames.
- Embedding tests confirming different crops from the same frame do not collapse trivially.
- Label-scoring tests on crops.
- Re-id graph tests for same-shot and cross-shot cases.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- SAM3 extraction quality is insufficient on some clips.
- Re-id thresholds over-merge or over-split.
- Developers quietly fall back to whole-scene evidence because it is easier.

## Questions the agent should ask before merging changes

- Are embeddings and labels definitely running on crops rather than full scenes?
- Can we point to the evidence for a source ID?
- Are ambiguous identities preserved instead of being forced?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means there is a trustworthy visual identity layer. The system can point to actual crops and say, “this is the visual evidence behind source X.”
