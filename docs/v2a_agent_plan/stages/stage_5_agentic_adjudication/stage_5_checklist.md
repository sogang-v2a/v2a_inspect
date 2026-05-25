# Stage 5 Checklist

[Back to stage main](stage_5_main.md) · [Read detailed plan](stage_5_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Agent core

- [x] Create planner state.
- [x] Create executor layer.
- [x] Create issue queue or equivalent ambiguity queue.
- [x] Implement bounded retry policies.

## Tool surface

- [x] Expose direct structural tools.
- [x] Expose direct extraction and recovery tools.
- [x] Expose direct crop, embedding, and label tools.
- [x] Expose direct grouping and routing-prior tools.
- [x] Expose direct validators.

## Logging

- [x] Log every tool call with inputs and output refs.
- [x] Log every merge/split acceptance decision.
- [x] Support replay of agent traces for debugging.

## Tests

- [x] Add planner tests.
- [x] Add executor tests.
- [x] Add budget enforcement tests.
- [x] Add trace replay tests.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
