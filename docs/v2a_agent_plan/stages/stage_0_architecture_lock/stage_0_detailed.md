# Stage 0 Detailed Plan: Architecture Lock, Semantic Invariants, and Gold Set

[Back to stage main](stage_0_main.md) · [Go to checklist](stage_0_checklist.md)

## Intended result

By the end of Stage 0, the project should have crossed a clear boundary: Done means there is one source of truth for what the final project is trying to build. If two developers describe the end-state differently after this stage, the stage is not actually done.

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- The current `use-tools` branch and its migration docs.
- The project goals from the track deck: multitrack description generation, sound grouping, TTA/VTA routing, validation/re-generation loops, and dataset construction.
- The decision to keep Gemini in the loop, but demote it from first-pass extractor to adjudicator.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- A frozen target bundle schema.
- A frozen tool contract for the agent.
- A current-to-target migration map.
- A gold evaluation set of short silent clips with human notes.
- A definition of stage exit criteria for the rest of the roadmap.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Lock the vocabulary

Define the three-layer semantics in plain language first: a physical source track is about identity, a sound event segment is about what that source is doing over time, and a generation group is about which segments can share one canonical sound recipe. This vocabulary must appear in the code, docs, tests, and UI. Avoid synonyms that invite confusion such as using “group” to mean both identity and acoustic equivalence.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Create the gold set

Pick a small but diverse silent-video set that includes single-source actions, crowds, repeated actors across cuts, moving vehicles, clear ambient scenes, and ambiguous cases. The gold set does not need perfect frame-level annotations, but it must include human notes on expected cuts, visible sound-producing sources, main events, and whether downstream routing should likely prefer TTA or VTA.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Write the migration map

Document which current artifacts survive, which become adapters, and which are deprecated. For example: current `SceneBoundary` becomes an early structural hint rather than the primary unit; `Sam3TrackSet` becomes an intermediate extraction artifact rather than a final track representation; `TrackGroup` splits into at least a source layer and a generation-group layer.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Define non-negotiable invariants

Write down rules that later code cannot violate without explicitly updating the blueprint. Examples: do not use audio; do not run embeddings on uncropped full-scene frames; do not merge cross-scene candidates silently when confidence is low; do not ask Gemini to do first-pass whole-video structural extraction in the final architecture.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Create `docs/adr/ADR_001_target_multitrack_architecture.md` or equivalent.
- Create `src/v2a_inspect/contracts/` placeholders even if many are empty at first.
- Create `tests/fixtures/gold_set/README.md` that lists sample clip IDs and expectations.
- Add a short `docs/current_to_target_mapping.md` or include that mapping in the roadmap overview.

## Implementation notes for the current repository

- Use docs and contract skeletons first. This stage should minimally disturb runtime code.
- Introduce new contract names now, even if adapters still point to old objects for a while.
- The gold set should include a small README with expected visible sources and grouping notes.
- Write examples that explicitly show: same source / different event, and different source / same generation group.
- Make the migration note blunt. If a current object has overloaded meaning, say so directly.

## What must not happen in this stage

- No major runtime refactor yet.
- No model switching experiments.
- No UI redesign.
- No benchmarking beyond establishing the gold set and expected artifacts.

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

- Docs-only lint or CI presence checks for architecture files.
- Validation that contract models import without heavy runtime dependencies.
- A smoke test that loads the gold set manifest.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- Team keeps using old names in new code and drifts back into ambiguity.
- The gold set is too easy and fails to expose the hard cases.
- People try to skip this stage because it feels non-technical.

## Questions the agent should ask before merging changes

- Did we actually freeze the semantics, or did we only describe them loosely?
- Could a new contributor tell the difference between source identity and generation grouping after reading the docs?
- Is the gold set diverse enough to catch the mistakes we are worried about?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means there is one source of truth for what the final project is trying to build. If two developers describe the end-state differently after this stage, the stage is not actually done.
