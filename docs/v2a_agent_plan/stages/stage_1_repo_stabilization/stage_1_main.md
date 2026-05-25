# Stage 1: Repository Stabilization and Testable Foundations

**Suggested duration:** 4-6 focused days

## Goal

Make the repository safe to iterate on. This stage reduces accidental breakage, disentangles imports, restores the ability to run lightweight tests, and creates a stable base for larger architectural work.

## Why this stage exists

The current branch already contains valuable code, but import-time coupling and heavy runtime assumptions make fast iteration difficult. If the team tries to add major new features before fixing the repo’s foundations, every step will be slower and debugging will be less trustworthy.

## Inputs

- The locked blueprint and contracts from Stage 0.
- The current client/server split already present in the repository.
- The current tests, including the known import-time dependency issues.

## Expected outputs

- Lazy or safe imports for top-level packages.
- A lightweight test path that does not require every heavy dependency at import time.
- A fake-tooling runtime for unit tests.
- Clear settings separation between client-safe logic and server-only logic.

## Primary code areas

- `src/v2a_inspect/__init__.py`.
- `server/src/v2a_inspect_server/__init__.py`.
- `src/v2a_inspect/runtime.py` and settings usage.
- The current tests in `tests/` and `server/tests/`.

## Explicitly out of scope

- No major algorithm redesign yet.
- No direct agent tool-calling yet.
- No new visual models.

## Main workstreams

### 1. Reduce import-time coupling

Move or wrap top-level imports so that importing `v2a_inspect` or `v2a_inspect_server` does not immediately import the heaviest runtime code. The rule is simple: contracts, tool types, validation utilities, and planner logic should be importable without pulling in Gemini, Transformers, or model weights.

### 2. Create fake runtimes and fixtures

Add simple fake implementations of tool outputs so higher-level pipeline code can be tested without GPUs. The fake runtime should return deterministic cuts, sampled frames, labels, embeddings, and route hints. It should be good enough to test orchestration, contracts, adapters, and validators.

### 3. Repair test drift

Update or rewrite tests that no longer match current function signatures. Avoid preserving stale tests just because they existed before. The new test suite should reflect the locked target semantics from Stage 0 and the repository boundaries defined here.

### 4. Harden settings and environment boundaries

Make it obvious which settings belong to the client/UI and which belong to the remote inference server. Server-only settings must not leak into modules that should remain lightweight. This matters later when the agent planner imports contract and validator code.


## Exit criteria

- Importing shared contracts and tool types no longer requires heavy Gemini or Transformers dependencies.
- Lightweight unit tests run in a clean environment.
- Server-only model loading is isolated to explicit runtime paths.
- There is a fake runtime path for contract, planner, and validator tests.

## How to use this stage folder

1. Read [the detailed stage plan](stage_1_detailed.md).
2. Use [the stage checklist](stage_1_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means the repo is easier to trust. New work can be unit-tested without a GPU, and import-time failures are no longer the main bottleneck.
