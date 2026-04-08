# Stage 4 Checklist

[Back to stage main](stage_4_main.md) · [Read detailed plan](stage_4_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Contracts

- [ ] Add `PhysicalSourceTrack`.
- [ ] Add `SoundEventSegment`.
- [ ] Add `AmbienceBed`.
- [ ] Add `GenerationGroup`.

## Semantics

- [ ] Define event-splitting rules.
- [ ] Define ambience criteria.
- [ ] Define acoustic-equivalence grouping rules.
- [ ] Document examples of same-identity/different-group and different-identity/same-group.

## Migration

- [ ] Add adapters to the legacy export shape.
- [ ] Document deprecation status of `RawTrack` and `TrackGroup`.

## Tests

- [ ] Add schema tests.
- [ ] Add adapter tests.
- [ ] Add event-splitting tests.
- [ ] Add grouping-semantic tests.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
