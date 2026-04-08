# Stage 3: Crop-Based Source Extraction, Re-Identification, and Identity Confidence

**Suggested duration:** 1-2 focused weeks

## Goal

Turn the current scene-local extraction hints into a real source-evidence layer. This stage implements crop-based track evidence, re-identification signals, label scoring, and stable source identities that later event logic can trust.

## Why this stage exists

Today the repository computes much of its embedding and label evidence on full-scene frames, which weakens the meaning of the visual tools. The end-state requires actual per-track crop evidence and stable identities. Without this stage, the agent is only pretending to use visual tools.

## Inputs

- Evidence windows from Stage 2.
- SAM3, DINOv2, and SigLIP2 runtimes already introduced on the server side.
- The contract rule that manual text-conditioned extraction is recovery-only at the external interface.

## Expected outputs

- Track crops generated from masks or boxes.
- Stable source-track IDs and identity confidence.
- DINOv2 embeddings computed on crops.
- SigLIP2 label scores computed on crops.
- A re-identification graph across windows and cuts.

## Primary code areas

- `server/src/v2a_inspect_server/sam3.py`.
- New crop-generation and re-id modules.
- `server/src/v2a_inspect_server/embeddings.py`.
- Shared tool contracts for track crops and identity edges.

## Explicitly out of scope

- No final event segmentation yet.
- No final generation groups yet.
- No full agentic tool loop yet.

## Main workstreams

### 1. Make crop generation real

Implement `crop_tracks` as a first-class tool. Crops should be created from masks when possible and from boxes with padding when masks are missing. Store both the original frame reference and the crop artifact. Later stages must be able to inspect which visual evidence supported a source track.

### 2. Define the external prompt-free contract

The blueprint says default extraction should be prompt-free. If the chosen SAM runtime needs prompts internally, hide that detail behind an external contract that does not require the planner or reviewer to hand-author text prompts. Internal seed prompts may come from SigLIP2 label suggestions or a small default vocabulary, but manual text recovery must remain a separate tool path.

### 3. Build re-identification as a graph, not a one-shot cluster

Use DINOv2 crop embeddings, temporal adjacency, label compatibility, and window continuity to create an identity graph. Same-window duplicates should be hard to merge; cross-window and cross-cut edges should require stronger evidence. Keep confidence on every identity edge so the agent can later inspect ambiguous merges.

### 4. Expose ambiguity

Not every visible source will be easy to track. When identity is weak, store that weakness explicitly rather than silently forcing a merge. A low-confidence source split is often safer than an unjustified merge, especially before generation grouping.


## Exit criteria

- Embeddings and labels are computed on actual crops, not raw whole-scene frames.
- Source IDs are stable enough to support later event splitting.
- The extraction layer exposes confidence and ambiguity explicitly.
- Recovery-only text prompting is separated from the default extraction interface.

## How to use this stage folder

1. Read [the detailed stage plan](stage_3_detailed.md).
2. Use [the stage checklist](stage_3_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means there is a trustworthy visual identity layer. The system can point to actual crops and say, “this is the visual evidence behind source X.”
