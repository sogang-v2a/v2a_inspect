# Stage 0 Checklist

[Back to stage main](stage_0_main.md) · [Read detailed plan](stage_0_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Architecture

- [ ] Write and review the target end-state blueprint.
- [ ] Write and review the agent tool contract v2.
- [ ] Freeze the definitions of `PhysicalSourceTrack`, `SoundEventSegment`, and `GenerationGroup`.
- [ ] Write a migration note for current `RawTrack` and `TrackGroup`.
- [ ] Record the non-goals and forbidden regressions.

## Gold set

- [ ] Create a manifest of representative silent clips.
- [ ] Add human notes for expected visible sources per clip.
- [ ] Add human notes for expected event boundaries per clip.
- [ ] Add human notes for likely grouping behavior.
- [ ] Add human notes for likely TTA/VTA preference.

## Code skeleton

- [ ] Create contract module placeholders.
- [ ] Ensure contract modules import without GPU dependencies.
- [ ] Add a fixture manifest loader test.

## Review

- [ ] Confirm the team will not start Stage 2 before this stage is signed off.
- [ ] Link the blueprint and roadmap from a top-level README.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
