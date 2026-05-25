# Stage 7 Checklist

[Back to stage main](stage_7_main.md) · [Read detailed plan](stage_7_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Dataset

- [x] Define the exported dataset record schema.
- [x] Store pipeline version and model versions.
- [x] Store review metadata and validation results.
- [x] Export evidence references or artifacts.

## Evaluation

- [x] Implement structural metrics.
- [x] Implement downstream generation experiment hooks.
- [ ] Run TTA-only, VTA-only, legacy, tool-only, and agentic baselines.
- [ ] Run at least one ablation on crop evidence.

## Demo and reporting

- [x] Create a reproducible demo script or guide.
- [ ] Collect qualitative examples and failure cases.
- [ ] Prepare tables and figures for report or paper.

## Tests

- [x] Add export tests.
- [x] Add evaluation harness smoke tests.
- [x] Add batch-processing smoke tests.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
