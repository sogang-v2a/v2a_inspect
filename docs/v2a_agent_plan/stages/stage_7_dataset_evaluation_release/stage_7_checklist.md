# Stage 7 Checklist

[Back to stage main](stage_7_main.md) · [Read detailed plan](stage_7_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Dataset

- [ ] Define the exported dataset record schema.
- [ ] Store pipeline version and model versions.
- [ ] Store review metadata and validation results.
- [ ] Export evidence references or artifacts.

## Evaluation

- [ ] Implement structural metrics.
- [ ] Implement downstream generation experiment hooks.
- [ ] Run TTA-only, VTA-only, legacy, tool-only, and agentic baselines.
- [ ] Run at least one ablation on crop evidence.

## Demo and reporting

- [ ] Create a reproducible demo script or guide.
- [ ] Collect qualitative examples and failure cases.
- [ ] Prepare tables and figures for report or paper.

## Tests

- [ ] Add export tests.
- [ ] Add evaluation harness smoke tests.
- [ ] Add batch-processing smoke tests.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
