# Stage 3 Checklist

[Back to stage main](stage_3_main.md) · [Read detailed plan](stage_3_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Extraction and crops

- [ ] Implement real crop generation from masks or boxes.
- [ ] Store crop artifacts with stable IDs.
- [ ] Retain source-to-frame provenance.

## Embeddings and labels

- [ ] Run DINOv2 on crops only.
- [ ] Run SigLIP2 on crops only.
- [ ] Store multiple label candidates with scores.

## Identity

- [ ] Implement a provisional re-id graph or equivalent source-track builder.
- [ ] Store identity confidence.
- [ ] Prevent same-window over-merging with stricter thresholds.
- [ ] Preserve unresolved ambiguity instead of forcing merges.

## Recovery path

- [ ] Separate manual text recovery from default extraction.
- [ ] Document when recovery may be invoked.

## Tests

- [ ] Add crop tests.
- [ ] Add re-id tests.
- [ ] Add crop-based embedding tests.
- [ ] Add label-scoring tests.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
