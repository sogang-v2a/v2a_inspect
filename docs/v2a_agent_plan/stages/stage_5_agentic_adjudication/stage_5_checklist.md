# Stage 5 Checklist

[Back to stage main](stage_5_main.md) · [Read detailed plan](stage_5_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Agent core

- [ ] Create planner state.
- [ ] Create executor layer.
- [ ] Create issue queue or equivalent ambiguity queue.
- [ ] Implement bounded retry policies.

## Tool surface

- [ ] Expose direct structural tools.
- [ ] Expose direct extraction and recovery tools.
- [ ] Expose direct crop, embedding, and label tools.
- [ ] Expose direct grouping and routing-prior tools.
- [ ] Expose direct validators.

## Logging

- [ ] Log every tool call with inputs and output refs.
- [ ] Log every merge/split acceptance decision.
- [ ] Support replay of agent traces for debugging.

## Tests

- [ ] Add planner tests.
- [ ] Add executor tests.
- [ ] Add budget enforcement tests.
- [ ] Add trace replay tests.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
