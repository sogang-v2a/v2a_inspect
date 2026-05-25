# Stage 4 Checklist

[Back to stage main](stage_4_main.md) · [Read detailed plan](stage_4_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Contracts

- [x] Add `PhysicalSourceTrack`.
- [x] Add `SoundEventSegment`.
- [x] Add `AmbienceBed`.
- [x] Add `GenerationGroup`.

## Semantics

- [x] Define event-splitting rules.
- [x] Define ambience criteria.
- [x] Define acoustic-equivalence grouping rules.
- [x] Document examples of same-identity/different-group and different-identity/same-group.

## Migration

- [x] Add adapters to the legacy export shape.
- [x] Document deprecation status of `RawTrack` and `TrackGroup`.

## Tests

- [x] Add schema tests.
- [x] Add adapter tests.
- [x] Add event-splitting tests.
- [x] Add grouping-semantic tests.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
