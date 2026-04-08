# Stage 3 Checklist

[Back to stage main](stage_3_main.md) · [Read detailed plan](stage_3_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Extraction and crops

- [x] Implement real crop generation from masks or boxes.
- [x] Store crop artifacts with stable IDs.
- [x] Retain source-to-frame provenance.

## Embeddings and labels

- [x] Run DINOv2 on crops only.
- [x] Run SigLIP2 on crops only.
- [x] Store multiple label candidates with scores.

## Identity

- [x] Implement a provisional re-id graph or equivalent source-track builder.
- [x] Store identity confidence.
- [x] Prevent same-window over-merging with stricter thresholds.
- [x] Preserve unresolved ambiguity instead of forcing merges.

## Recovery path

- [x] Separate manual text recovery from default extraction.
- [x] Document when recovery may be invoked.

## Tests

- [x] Add crop tests.
- [x] Add re-id tests.
- [x] Add crop-based embedding tests.
- [x] Add label-scoring tests.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
