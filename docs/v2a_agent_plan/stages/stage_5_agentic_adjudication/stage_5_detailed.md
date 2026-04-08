# Stage 5 Detailed Plan: Direct Agentic Tool Use, Planner/Executor, and Bounded Repair Loops

[Back to stage main](stage_5_main.md) · [Go to checklist](stage_5_checklist.md)

## Intended result

By the end of Stage 5, the project should have crossed a clear boundary: Done means the system now deserves to be called agentic: the reasoner can actually choose and call tools, not just read summaries of tools that someone else already ran.

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- Stable structural, source, and event layers from Stages 2-4.
- The direct tool surface defined in the blueprint.
- The current server runtime as the place where heavy tools execute.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- A planner/executor layer.
- Direct tool endpoints or callable adapters.
- Bounded retry and ambiguity-resolution logic.
- Action logs for every agent decision.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Build the planner/executor split

The planner decides which unresolved issue to tackle next and which tool to call. The executor handles the actual call, serialization, artifact paths, and retries. This separation keeps the agent readable and easier to test. Do not collapse both into one giant prompt-orchestration function.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Expose a direct tool registry

Every important visual tool should have a typed request/response interface. This can be in-process Python callables, HTTP endpoints, or both, but the interface must be explicit. The agent should be able to request: structural evidence, extraction, crop generation, embedding, labeling, grouping proposals, routing priors, validation checks, and recovery passes.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Add bounded loops and budgets

This is crucial. The agent may be allowed to retry extraction or regrouping, but only under clear limits. Examples: at most one manual recovery prompt per ambiguous source candidate; at most two regroup attempts for one ambiguity cluster; at most three global validation-repair rounds per video. Without this, the system will become expensive and hard to debug.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Log every decision

Each agent decision should store the unresolved item, the chosen tool, the returned artifact IDs, and the rationale for accepting or rejecting a merge/split. This is important for debugging, UI review, and later research writing.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Create `src/v2a_inspect/agent/state.py`, `planner.py`, `executor.py`, and `policies.py` or equivalent.
- Create a server-side tool registry and direct request handlers.
- Keep the current tool-context hint builder only as a temporary compatibility adapter; do not make it the final path.

## Implementation notes for the current repository

- Keep the current `tool_context`-as-hints flow alive only as a compatibility path while the agentic path is introduced.
- The planner should work from typed unresolved issues or ambiguity items rather than free-form prompt text.
- The executor should own tool-call serialization, budgets, and logging so the planner remains readable.
- Add replayable action logs early so debugging the first agentic runs is possible.
- Resist the urge to let the agent call arbitrary tools without a typed registry and cost policy.

## What must not happen in this stage

- No full dataset export yet.
- No final paper evaluation yet.
- No speculative multi-provider abstraction.

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

- Planner tests with fake unresolved items.
- Executor tests with fake tool outputs.
- Budget enforcement tests.
- Replay/logging tests for action traces.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- The planner becomes an unbounded chain-of-thought sandbox with no cost control.
- Developers keep relying on the old hint-append pattern because it is familiar.
- The tool registry becomes inconsistent or weakly typed.

## Questions the agent should ask before merging changes

- Can the agent truly call tools directly, or are we still mostly appending hint text to prompts?
- Are tool-call budgets enforced in code?
- Can a bad or confusing run be replayed from logs?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means the system now deserves to be called agentic: the reasoner can actually choose and call tools, not just read summaries of tools that someone else already ran.
