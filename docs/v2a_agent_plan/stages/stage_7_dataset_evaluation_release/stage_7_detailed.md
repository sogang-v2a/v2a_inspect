# Stage 7 Detailed Plan: Dataset Builder, Evaluation, Demo Hardening, and Research Packaging

[Back to stage main](stage_7_main.md) · [Go to checklist](stage_7_checklist.md)

## Intended result

By the end of Stage 7, the project should have crossed a clear boundary: Done means the project can produce reproducible artifacts, defend its design decisions with data, and support a final demo or paper submission.

This stage should be implemented in a way that keeps the final blueprint intact. Do not treat this stage as an excuse to redesign the target architecture informally.

## Prerequisite check

Before coding, confirm the following inputs really exist and are not only assumed:

- Final bundle generation from Stage 6.
- The gold set and validation logic from earlier stages.
- The project requirement to compare strategies such as TTA-only, VTA-only, routing, and validation/repair loops.

If one of these inputs is missing, pause and repair the earlier stage instead of improvising around the gap.

## Deliverables to collect during the stage

- Dataset export format and batch builder.
- Evaluation harness and baseline runners.
- A hardened demo flow.
- Research artifacts: ablation tables, examples, and experiment manifests.

Treat these as artifacts to gather, not as vague intentions. At the end of the stage, someone should be able to point to concrete files, tests, or code paths that correspond to each item.

## Step-by-step implementation sequence

### Step 1: Define the dataset record

A processed example should include the video reference, the multitrack bundle, the evidence artifacts or references, review metadata, pipeline version, model versions, and validation results. Make this export stable before scaling out. A bad export shape will pollute everything downstream.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 2: Build the evaluation harness

Separate structural evaluation from downstream audio-generation evaluation. Structural metrics may include source coverage, event-boundary agreement, grouping accuracy, and routing agreement with human judgment. Downstream generation metrics can include FAD, KLD, CLAP, AV alignment, and human preference, but keep those as a second layer rather than the only measurement.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 3: Run baselines and ablations

At minimum compare: the current legacy Gemini-heavy path, the new tool-first path without the agent, the full agentic path, TTA-only routing, VTA-only routing, and the final routed path. Add ablations for crops vs. full-scene embeddings, direct tool-calling vs. prompt hints, and source-vs-generation-group semantics.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

### Step 4: Package the demo and report

Create a reproducible demo path, example videos, screenshots of evidence windows and group reviews, and a short explanation of the pipeline’s architectural contribution. This makes the project useful both internally and for final presentation or paper writing.

Practical rule for this step: implement the smallest version that satisfies the stage exit criteria, but do not undermine the final architecture by taking a shortcut that changes semantics.

## File-by-file plan

The exact filenames may vary, but the repository changes should stay close to the following plan.

- Add `dataset/` export modules and manifests.
- Add `evaluation/metrics.py`, `baselines.py`, and experiment configs.
- Document the demo flow and reproducibility steps.

## Implementation notes for the current repository

- Do not scale export until the bundle schema is stable.
- Separate structural evaluation from downstream audio-generation evaluation.
- Compare the new architecture against the legacy path and simplified baselines, not just against hand-picked demos.
- Preserve failure cases as first-class artifacts; they are important for research and debugging.
- Every exported example should include pipeline versioning and model versioning.

## What must not happen in this stage

- No endless benchmark expansion before the core benchmark is stable.
- No premature support for many deployment providers.

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

- Dataset export schema tests.
- Experiment manifest tests.
- Evaluation harness smoke tests.
- End-to-end batch processing smoke tests on a small sample.

## Evidence to save before closing the stage

Before marking the stage complete, collect a small set of evidence artifacts such as:

- example outputs from fixture videos
- screenshots or JSON snippets showing the new structure
- passing test output or a short run note
- a note describing any temporary adapter introduced in the stage

## Failure modes to watch for

- The team scales data export before the export format stabilizes.
- Baselines are not truly comparable.
- Evaluation focuses only on downstream audio metrics and ignores structural quality.

## Questions the agent should ask before merging changes

- Can we reproduce exported dataset records later?
- Do the baselines reflect genuine alternatives rather than strawmen?
- Did we keep structural metrics separate from downstream audio-generation metrics?

## Stage handoff requirements

Before moving to the next stage, update any affected docs, adapters, and tests. The next stage should begin with a stable foundation rather than a vague memory of what changed.

## Definition of done

Done means the project can produce reproducible artifacts, defend its design decisions with data, and support a final demo or paper submission.
