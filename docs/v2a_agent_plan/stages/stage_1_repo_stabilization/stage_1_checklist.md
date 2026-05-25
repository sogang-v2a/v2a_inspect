# Stage 1 Checklist

[Back to stage main](stage_1_main.md) · [Read detailed plan](stage_1_detailed.md)

Use this checklist as the stage exit gate. If a blocking item is still incomplete, the stage should not be considered finished.

## Imports and packaging

- [x] Remove eager heavy imports from client package root.
- [x] Remove eager heavy imports from server package root.
- [x] Verify shared contract modules import cleanly.
- [x] Verify tool type modules import cleanly.

## Tests

- [x] Add fake runtime objects for orchestration tests.
- [x] Fix current stale tests to match actual signatures.
- [x] Run unit tests that do not depend on model weights.
- [x] Add at least one smoke test for the current server request flow using fakes.

## Settings and boundaries

- [x] Separate client-safe settings from server-only settings where needed.
- [x] Ensure server-only paths are only touched by explicit runtime code.
- [x] Document required environment variables per runtime mode.

## Review

- [x] Confirm the repo can be worked on without full remote GPU availability for every change.
- [x] Document the remaining heavy-dependency integration tests that still require the real server.

## Final sign-off

- [x] Stage outputs are linked from the top-level README or docs index.
- [x] New or changed tests have been run.
- [x] Temporary adapters are labeled as temporary.
- [x] The next stage can name its inputs concretely rather than vaguely.
