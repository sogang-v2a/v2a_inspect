# Stage 4: Event Segmentation, Ambience Beds, and Generation-Group Semantics

**Suggested duration:** 1 focused week

## Goal

Build the semantic middle layer that the current system lacks. This stage converts source identity into event segments and explicitly separates those event segments from the final generation groups used for canonical sound descriptions and downstream routing.

## Why this stage exists

The current pipeline overloads one grouping object with two jobs: identity and acoustic equivalence. That makes the behavior unstable. This stage fixes the ontology so future routing and validation can behave sensibly.

## Inputs

- Stable source evidence from Stage 3.
- The three-layer schema locked in Stage 0.
- The project requirement that grouping should be about acoustic similarity, not necessarily physical identity.

## Expected outputs

- Physical source tracks.
- Sound event segments derived from those source tracks.
- Ambience beds as a separate class of output.
- Generation-group proposals based on acoustic equivalence.

## Primary code areas

- New contract modules under `src/v2a_inspect/contracts/`.
- New event-splitting logic on the server or shared pipeline layer.
- Adapters that still let the old UI or export path read the new structures during migration.

## Explicitly out of scope

- No direct agent tool-calling yet.
- No final validation loop yet.
- No dataset builder yet.

## Main workstreams

### 1. Promote source tracks to first-class contracts

Create real `PhysicalSourceTrack` contracts that can survive across windows and cuts. A source track should know its evidence, spans, label candidates, confidence, and whether it is foreground or background-like. It should not yet claim to be a generation prompt.

### 2. Split sources into event segments

A source can persist while its sound changes. Use motion regime changes, interaction/contact changes, start/stop boundaries, and material/context changes to split a source span into `SoundEventSegment`s. This is where the project’s desired shift from object-centric analysis toward event-aware analysis becomes concrete.

### 3. Represent ambience beds explicitly

Background ambience should not be a trash bin for excess objects. Represent ambience or environmental beds separately so later routing and generation decisions can treat them differently from discrete event segments. Ambience should still be visually justified; do not invent invisible soundtrack layers.

### 4. Build generation groups from acoustic equivalence

Generation groups should be about whether two event segments can share one canonical sound recipe. A person running and another person running on the same material may belong to the same generation group even though they are not the same physical source. Conversely, the same person walking on gravel and then stepping onto metal should likely be split.


## Exit criteria

- The repository can represent source identity separately from generation grouping.
- Ambience is no longer treated as just another object track.
- Event segments can express changing acoustic context within one persistent source.
- Adapters exist so old result rendering can survive temporarily while migration continues.

## How to use this stage folder

1. Read [the detailed stage plan](stage_4_detailed.md).
2. Use [the stage checklist](stage_4_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means the pipeline can finally say three different things cleanly: who the source is, what the source is doing now, and which other segments can share one canonical generated sound description.
