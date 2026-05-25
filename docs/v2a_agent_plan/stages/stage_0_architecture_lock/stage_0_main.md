# Stage 0: Architecture Lock, Semantic Invariants, and Gold Set

**Suggested duration:** 2-3 focused days

## Goal

Freeze the target end-state before more code is written. The main deliverable of this stage is not a flashy feature; it is a stable set of meanings, contracts, and evaluation fixtures so the agent stops drifting back toward the current scene-first Gemini-heavy design.

## Why this stage exists

The current repository mixes several incompatible concepts: Gemini scenes, SAM3 scene-local tracks, and grouped sound descriptions. Without a locked target schema, future work will keep reintroducing the same confusion. This stage creates one authoritative definition of the end-state and a small gold set that every later stage must respect.

## Inputs

- The current `use-tools` branch and its migration docs.
- The project goals from the track deck: multitrack description generation, sound grouping, TTA/VTA routing, validation/re-generation loops, and dataset construction.
- The decision to keep Gemini in the loop, but demote it from first-pass extractor to adjudicator.

## Expected outputs

- A frozen target bundle schema.
- A frozen tool contract for the agent.
- A current-to-target migration map.
- A gold evaluation set of short silent clips with human notes.
- A definition of stage exit criteria for the rest of the roadmap.

## Primary code areas

- `docs/` for ADRs and contracts.
- New `contracts/` package skeleton in `src/v2a_inspect/`.
- A small `tests/fixtures/gold_set/` area or equivalent.

## Explicitly out of scope

- No major runtime refactor yet.
- No model switching experiments.
- No UI redesign.
- No benchmarking beyond establishing the gold set and expected artifacts.

## Main workstreams

### 1. Lock the vocabulary

Define the three-layer semantics in plain language first: a physical source track is about identity, a sound event segment is about what that source is doing over time, and a generation group is about which segments can share one canonical sound recipe. This vocabulary must appear in the code, docs, tests, and UI. Avoid synonyms that invite confusion such as using “group” to mean both identity and acoustic equivalence.

### 2. Create the gold set

Pick a small but diverse silent-video set that includes single-source actions, crowds, repeated actors across cuts, moving vehicles, clear ambient scenes, and ambiguous cases. The gold set does not need perfect frame-level annotations, but it must include human notes on expected cuts, visible sound-producing sources, main events, and whether downstream routing should likely prefer TTA or VTA.

### 3. Write the migration map

Document which current artifacts survive, which become adapters, and which are deprecated. For example: current `SceneBoundary` becomes an early structural hint rather than the primary unit; `Sam3TrackSet` becomes an intermediate extraction artifact rather than a final track representation; `TrackGroup` splits into at least a source layer and a generation-group layer.

### 4. Define non-negotiable invariants

Write down rules that later code cannot violate without explicitly updating the blueprint. Examples: do not use audio; do not run embeddings on uncropped full-scene frames; do not merge cross-scene candidates silently when confidence is low; do not ask Gemini to do first-pass whole-video structural extraction in the final architecture.


## Exit criteria

- The team agrees on the distinction between physical source tracks, sound event segments, and generation groups.
- The target bundle schema exists in code or pseudo-code and is referenced by later stages.
- The gold set is checked in or otherwise frozen with filenames and notes.
- A migration table exists that explains how current `RawTrack`, `TrackGroup`, and tool objects map to the final model.

## How to use this stage folder

1. Read [the detailed stage plan](stage_0_detailed.md).
2. Use [the stage checklist](stage_0_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means there is one source of truth for what the final project is trying to build. If two developers describe the end-state differently after this stage, the stage is not actually done.
