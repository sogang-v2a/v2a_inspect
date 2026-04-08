# Stage 0 Checklist

[Back to stage main](stage_0_main.md) · [Read detailed plan](stage_0_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Architecture

- [x] Write and review the target end-state blueprint.
- [x] Write and review the agent tool contract v2.
- [x] Freeze the definitions of `PhysicalSourceTrack`, `SoundEventSegment`, and `GenerationGroup`.
- [x] Write a migration note for current `RawTrack` and `TrackGroup`.
- [x] Record the non-goals and forbidden regressions.

## Gold set

- [x] Create a manifest of representative silent clips.
- [x] Add human notes for expected visible sources per clip.
- [x] Add human notes for expected event boundaries per clip.
- [x] Add human notes for likely grouping behavior.
- [x] Add human notes for likely TTA/VTA preference.

## Code skeleton

- [x] Create contract module placeholders.
- [x] Ensure contract modules import without GPU dependencies.
- [x] Add a fixture manifest loader test.

## Review

- [x] Confirm the team will not start Stage 2 before this stage is signed off.
- [x] Link the blueprint and roadmap from a top-level README.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
