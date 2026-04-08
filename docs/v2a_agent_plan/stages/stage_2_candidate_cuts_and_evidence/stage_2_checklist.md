# Stage 2 Checklist

[Back to stage main](stage_2_main.md) · [Read detailed plan](stage_2_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Contracts

- [ ] Add `CandidateCut` contract model.
- [ ] Add `EvidenceWindow` contract model.
- [ ] Add typed cut reasons and confidence fields.

## Algorithms

- [ ] Implement hard shot-boundary proposals.
- [ ] Implement soft cut proposals based on simple structural cues.
- [ ] Implement duplicate/near-duplicate cut merging.
- [ ] Implement minimum-window-length handling.

## Artifacts

- [ ] Generate sampled frames per evidence window.
- [ ] Generate storyboard artifacts.
- [ ] Support optional short visual clip export.

## Tests

- [ ] Add fixture tests for cut proposals.
- [ ] Add tests for merge rules and edge cases.
- [ ] Add storyboard export tests.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
