# Stage 6 Checklist

[Back to stage main](stage_6_main.md) · [Read detailed plan](stage_6_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Descriptions

- [x] Generate canonical descriptions from structured evidence.
- [x] Store rationale and confidence for each description.
- [x] Separate ambience descriptions from discrete-event descriptions.

## Routing

- [x] Implement deterministic routing priors.
- [x] Implement adjudicated final route decisions.
- [x] Deprecate old threshold-only routing as the main path.

## Validation

- [x] Add typed validation issues.
- [x] Map issues to targeted reruns or review actions.
- [x] Persist validation results in the final bundle.

## UI

- [x] Render evidence windows and crops.
- [x] Render source tracks, event segments, and generation groups.
- [x] Support route overrides.
- [x] Support split/merge corrections.
- [x] Persist edits.

## Tests

- [x] Add routing tests.
- [x] Add validator tests.
- [x] Add bundle export tests.
- [x] Add UI edit-state tests where feasible.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
