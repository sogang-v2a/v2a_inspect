# Stage 2 Checklist

[Back to stage main](stage_2_main.md) · [Read detailed plan](stage_2_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Contracts

- [x] Add `CandidateCut` contract model.
- [x] Add `EvidenceWindow` contract model.
- [x] Add typed cut reasons and confidence fields.

## Algorithms

- [x] Implement hard shot-boundary proposals.
- [x] Implement soft cut proposals based on simple structural cues.
- [x] Implement duplicate/near-duplicate cut merging.
- [x] Implement minimum-window-length handling.

## Artifacts

- [x] Generate sampled frames per evidence window.
- [x] Generate storyboard artifacts.
- [x] Support optional short visual clip export.

## Tests

- [x] Add fixture tests for cut proposals.
- [x] Add tests for merge rules and edge cases.
- [x] Add storyboard export tests.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
