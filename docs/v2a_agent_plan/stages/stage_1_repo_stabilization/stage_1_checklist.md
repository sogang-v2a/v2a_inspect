# Stage 1 Checklist

[Back to stage main](stage_1_main.md) · [Read detailed plan](stage_1_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Imports and packaging

- [ ] Remove eager heavy imports from client package root.
- [ ] Remove eager heavy imports from server package root.
- [ ] Verify shared contract modules import cleanly.
- [ ] Verify tool type modules import cleanly.

## Tests

- [ ] Add fake runtime objects for orchestration tests.
- [ ] Fix current stale tests to match actual signatures.
- [ ] Run unit tests that do not depend on model weights.
- [ ] Add at least one smoke test for the current server request flow using fakes.

## Settings and boundaries

- [ ] Separate client-safe settings from server-only settings where needed.
- [ ] Ensure server-only paths are only touched by explicit runtime code.
- [ ] Document required environment variables per runtime mode.

## Review

- [ ] Confirm the repo can be worked on without full remote GPU availability for every change.
- [ ] Document the remaining heavy-dependency integration tests that still require the real server.

## Final sign-off

- [ ] Stage outputs are linked from the top-level README or docs index.
- [ ] New or changed tests have been run.
- [ ] Temporary adapters are labeled as temporary.
- [ ] The next stage can name its inputs concretely rather than vaguely.
