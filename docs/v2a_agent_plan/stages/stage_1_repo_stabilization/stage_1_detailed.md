# [Historical note] This planning document predates the silent-video proposal-stack redesign. References to legacy Gemini upload or tool_context are superseded by the active runtime in README and docs/tooling_constraints.md.

# Stage 1 Detailed Plan: Repository Stabilization and Testable Foundations

[Back to stage main](stage_1_main.md) · [Go to checklist](stage_1_checklist.md)

## Intended result

By the end of Stage 1, the project should have crossed a clear boundary: Done means the repo is easier to trust. New work can be unit-tested without a GPU, and import-time failures are no longer the main bottleneck.

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- The locked blueprint and contracts from Stage 0.
- The current client/server split already present in the repository.
- The current tests, including the known import-time dependency issues.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- Lazy or safe imports for top-level packages.
- A lightweight test path that does not require every heavy dependency at import time.
- A fake-tooling runtime for unit tests.
- Clear settings separation between client-safe logic and server-only logic.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Reduce import-time coupling

Move or wrap top-level imports so that importing `v2a_inspect` or `v2a_inspect_server` does not immediately import the heaviest runtime code. The rule is simple: contracts, tool types, validation utilities, and planner logic should be importable without pulling in Gemini, Transformers, or model weights.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Create fake runtimes and fixtures

Add simple fake implementations of tool outputs so higher-level pipeline code can be tested without GPUs. The fake runtime should return deterministic cuts, sampled frames, labels, embeddings, and route hints. It should be good enough to test orchestration, contracts, adapters, and validators.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Repair test drift

Update or rewrite tests that no longer match current function signatures. Avoid preserving stale tests just because they existed before. The new test suite should reflect the locked target semantics from Stage 0 and the repository boundaries defined here.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Harden settings and environment boundaries

Make it obvious which settings belong to the client/UI and which belong to the remote inference server. Server-only settings must not leak into modules that should remain lightweight. This matters later when the agent planner imports contract and validator code.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Refactor `src/v2a_inspect/__init__.py` to avoid eager runner/runtime imports.
- Refactor `server/src/v2a_inspect_server/__init__.py` to avoid eager model imports.
- Add `tests/fakes/` and `server/tests/fakes/` or equivalent.
- Split utility modules so shared schemas do not import runtime builders.
- Fix stale tests in `server/tests/test_tool_context.py` and similar files.

## Implementation notes for the current repository

- Refactor `src/v2a_inspect/__init__.py` so importing the package does not immediately construct Gemini runtime objects.
- Refactor `server/src/v2a_inspect_server/__init__.py` so importing the package does not immediately import Transformers-backed model loaders.
- Move code that truly requires heavy deps behind explicit runtime factory functions.
- Add fakes for tools and make planner/contract tests depend on those fakes instead of real server runtime.
- Fix stale tests rather than working around them; update signatures and assumptions to match the current code.
- Keep shared contracts and validators import-light so later agent code can use them freely.

## What must not happen in this stage

- No major algorithm redesign yet.
- No direct agent tool-calling yet.
- No new visual models.

In addition to the items above, do not silently introduce new semantics that belong to later stages.

## Suggested milestone breakdown inside the stage

A useful internal cadence for this stage is:

1. add or modify contracts and tests first
2. implement the minimal code path
3. run fixtures and inspect artifacts manually
4. only then refine thresholds or prompts
5. update docs and adapters before declaring the stage complete

## Tests and measurement for this stage

A stage is only meaningful if there is a way to tell whether it worked.

- Package import tests for client and server roots.
- Smoke tests for fake tool runtime outputs.
- Settings tests for client-safe and server-only separation.
- Minimal end-to-end orchestration smoke test using fakes.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- A developer reintroduces heavy imports through convenience exports.
- Tests still accidentally rely on local environment state.
- This stage gets rushed because it does not produce visible product features.

## Questions the agent should ask before merging changes

- Can lightweight tests run without GPU or external model downloads?
- Are imports still accidentally pulling in heavy runtime code?
- Did we fix stale tests or merely bypass them?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means the repo is easier to trust. New work can be unit-tested without a GPU, and import-time failures are no longer the main bottleneck.
