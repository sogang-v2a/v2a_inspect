# Stage 6 Checklist

[Back to stage main](stage_6_main.md) · [Read detailed plan](stage_6_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Descriptions

- [ ] Generate canonical descriptions from structured evidence.
- [ ] Store rationale and confidence for each description.
- [ ] Separate ambience descriptions from discrete-event descriptions.

## Routing

- [ ] Implement deterministic routing priors.
- [ ] Implement adjudicated final route decisions.
- [ ] Deprecate old threshold-only routing as the main path.

## Validation

- [ ] Add typed validation issues.
- [ ] Map issues to targeted reruns or review actions.
- [ ] Persist validation results in the final bundle.

## UI

- [ ] Render evidence windows and crops.
- [ ] Render source tracks, event segments, and generation groups.
- [ ] Support route overrides.
- [ ] Support split/merge corrections.
- [ ] Persist edits.

## Tests

- [ ] Add routing tests.
- [ ] Add validator tests.
- [ ] Add bundle export tests.
- [ ] Add UI edit-state tests where feasible.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
