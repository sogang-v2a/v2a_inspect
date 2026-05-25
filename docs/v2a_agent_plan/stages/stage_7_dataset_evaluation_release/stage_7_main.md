# Stage 7: Dataset Builder, Evaluation, Demo Hardening, and Research Packaging

**Suggested duration:** 1-2 focused weeks for first pass, then ongoing

## Goal

Turn the completed pipeline into a dataset-producing and research-ready system. This stage is where the project becomes reproducible, measurable, and presentable beyond the dev environment.

## Why this stage exists

Your stated project goals include dataset construction, validation, and paper submission. Those goals are only credible if the pipeline can export stable artifacts, run baselines, and report meaningful metrics. This stage institutionalizes that work.

## Inputs

- Final bundle generation from Stage 6.
- The gold set and validation logic from earlier stages.
- The project requirement to compare strategies such as TTA-only, VTA-only, routing, and validation/repair loops.

## Expected outputs

- Dataset export format and batch builder.
- Evaluation harness and baseline runners.
- A hardened demo flow.
- Research artifacts: ablation tables, examples, and experiment manifests.

## Primary code areas

- `evaluation/` and `dataset/` modules.
- Export scripts and manifests.
- Demo packaging and documentation.

## Explicitly out of scope

- No endless benchmark expansion before the core benchmark is stable.
- No premature support for many deployment providers.

## Main workstreams

### 1. Define the dataset record

A processed example should include the video reference, the multitrack bundle, the evidence artifacts or references, review metadata, pipeline version, model versions, and validation results. Make this export stable before scaling out. A bad export shape will pollute everything downstream.

### 2. Build the evaluation harness

Separate structural evaluation from downstream audio-generation evaluation. Structural metrics may include source coverage, event-boundary agreement, grouping accuracy, and routing agreement with human judgment. Downstream generation metrics can include FAD, KLD, CLAP, AV alignment, and human preference, but keep those as a second layer rather than the only measurement.

### 3. Run baselines and ablations

At minimum compare: the current legacy Gemini-heavy path, the new tool-first path without the agent, the full agentic path, TTA-only routing, VTA-only routing, and the final routed path. Add ablations for crops vs. full-scene embeddings, direct tool-calling vs. prompt hints, and source-vs-generation-group semantics.

### 4. Package the demo and report

Create a reproducible demo path, example videos, screenshots of evidence windows and group reviews, and a short explanation of the pipeline’s architectural contribution. This makes the project useful both internally and for final presentation or paper writing.


## Exit criteria

- The pipeline can export a stable dataset record per processed video.
- The evaluation harness can compare multiple routing/grouping strategies.
- The demo is reproducible and not dependent on hidden manual steps.
- The team has the tables and qualitative examples needed for a paper or final report.

## How to use this stage folder

1. Read [the detailed stage plan](stage_7_detailed.md).
2. Use [the stage checklist](stage_7_checklist.md) as the exit gate.
3. Do not start the next stage while blocking items here remain undone.

## End-of-stage meaning

Done means the project can produce reproducible artifacts, defend its design decisions with data, and support a final demo or paper submission.
