# Stage 4 Detailed Plan: Event Segmentation, Ambience Beds, and Generation-Group Semantics

[Back to stage main](stage_4_main.md) · [Go to checklist](stage_4_checklist.md)

## Intended result

By the end of Stage 4, the project should have crossed a clear boundary: Done means the pipeline can finally say three different things cleanly: who the source is, what the source is doing now, and which other segments can share one canonical generated sound description.

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- Stable source evidence from Stage 3.
- The three-layer schema locked in Stage 0.
- The project requirement that grouping should be about acoustic similarity, not necessarily physical identity.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- Physical source tracks.
- Sound event segments derived from those source tracks.
- Ambience beds as a separate class of output.
- Generation-group proposals based on acoustic equivalence.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Promote source tracks to first-class contracts

Create real `PhysicalSourceTrack` contracts that can survive across windows and cuts. A source track should know its evidence, spans, label candidates, confidence, and whether it is foreground or background-like. It should not yet claim to be a generation prompt.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Split sources into event segments

A source can persist while its sound changes. Use motion regime changes, interaction/contact changes, start/stop boundaries, and material/context changes to split a source span into `SoundEventSegment`s. This is where the project’s desired shift from object-centric analysis toward event-aware analysis becomes concrete.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Represent ambience beds explicitly

Background ambience should not be a trash bin for excess objects. Represent ambience or environmental beds separately so later routing and generation decisions can treat them differently from discrete event segments. Ambience should still be visually justified; do not invent invisible soundtrack layers.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Build generation groups from acoustic equivalence

Generation groups should be about whether two event segments can share one canonical sound recipe. A person running and another person running on the same material may belong to the same generation group even though they are not the same physical source. Conversely, the same person walking on gravel and then stepping onto metal should likely be split.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Add contracts such as `PhysicalSourceTrack`, `SoundEventSegment`, `AmbienceBed`, and `GenerationGroup`.
- Add adapter utilities that translate new structures into legacy `GroupedAnalysis` when needed.
- Create event-splitting logic and acoustic-equivalence grouping proposals.

## Implementation notes for the current repository

- Create the new contracts first, then write adapters from current legacy structures into the new ones.
- Do not let `GenerationGroup` become a renamed `TrackGroup`; it must represent acoustic equivalence rather than raw identity.
- Add explicit ambience objects rather than stuffing extra content into generic background text.
- Event splitting can begin with heuristics; it does not need to wait for a perfect learned event detector.
- Keep examples handy for confusing cases such as one source performing multiple acoustically distinct actions.

## What must not happen in this stage

- No direct agent tool-calling yet.
- No final validation loop yet.
- No dataset builder yet.

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

- Contract validation tests for the new schema.
- Adapter tests from new structures to legacy export.
- Event-splitting tests on representative fixture clips.
- Grouping tests that prove identity and generation groups are not the same concept.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- The team shortcuts the ontology and keeps using old grouping semantics.
- Ambience becomes overly broad again.
- Too many event segments are produced and the output becomes noisy.

## Questions the agent should ask before merging changes

- Can the code represent one source with multiple event segments?
- Can two different sources share one generation group without claiming they are the same entity?
- Is ambience treated explicitly?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means the pipeline can finally say three different things cleanly: who the source is, what the source is doing now, and which other segments can share one canonical generated sound description.
