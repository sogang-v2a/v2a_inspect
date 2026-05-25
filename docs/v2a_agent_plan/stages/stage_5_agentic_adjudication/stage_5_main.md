# [Historical note] This planning document predates the silent-video proposal-stack redesign. References to legacy Gemini upload or tool_context are superseded by the active runtime in README and docs/tooling_constraints.md.

# Stage 5: Direct Agentic Tool Use, Planner/Executor, and Bounded Repair Loops

**Suggested duration:** 1-2 focused weeks

## Goal

Replace the current “tool hints appended to Gemini prompts” pattern with a real agentic loop that can directly call tools, inspect outputs, request recoveries, and produce justified merge/split decisions under strict budgets.

## Why this stage exists

The current branch is a hybrid scaffold, not the final agentic system. Its visual tools exist, but Gemini mostly sees them as text hints. This stage is where the project actually becomes agentic in the sense you described.

## Inputs

- Stable structural, source, and event layers from Stages 2-4.
- The direct tool surface defined in the blueprint.
- The current server runtime as the place where heavy tools execute.

## Expected outputs

- A planner/executor layer.
- Direct tool endpoints or callable adapters.
- Bounded retry and ambiguity-resolution logic.
- Action logs for every agent decision.

## Primary code areas

- A new `agent/` package in `src/v2a_inspect/`.
- New tool registry and endpoints in `server/src/v2a_inspect_server/`.
- Replacement or adaptation of the current `tool_context`-hint flow.

## Explicitly out of scope

- No full dataset export yet.
- No final paper evaluation yet.
- No speculative multi-provider abstraction.

## Main workstreams

### 1. Build the planner/executor split

The planner decides which unresolved issue to tackle next and which tool to call. The executor handles the actual call, serialization, artifact paths, and retries. This separation keeps the agent readable and easier to test. Do not collapse both into one giant prompt-orchestration function.

### 2. Expose a direct tool registry

Every important visual tool should have a typed request/response interface. This can be in-process Python callables, HTTP endpoints, or both, but the interface must be explicit. The agent should be able to request: structural evidence, extraction, crop generation, embedding, labeling, grouping proposals, routing priors, validation checks, and recovery passes.

### 3. Add bounded loops and budgets

This is crucial. The agent may be allowed to retry extraction or regrouping, but only under clear limits. Examples: at most one manual recovery prompt per ambiguous source candidate; at most two regroup attempts for one ambiguity cluster; at most three global validation-repair rounds per video. Without this, the system will become expensive and hard to debug.

### 4. Log every decision

Each agent decision should store the unresolved item, the chosen tool, the returned artifact IDs, and the rationale for accepting or rejecting a merge/split. This is important for debugging, UI review, and later research writing.


## Exit criteria

- The agent can call tools iteratively instead of consuming only precomputed text hints.
- The agent can trigger targeted recoveries and not just whole-pipeline reruns.
- Tool calls and decisions are logged for replay and debugging.
- The loop is bounded and cannot wander indefinitely.

## How to use this stage folder

1. Read [the detailed stage plan](stage_5_detailed.md).
2. Use [the stage checklist](stage_5_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means the system now deserves to be called agentic: the reasoner can actually choose and call tools, not just read summaries of tools that someone else already ran.
