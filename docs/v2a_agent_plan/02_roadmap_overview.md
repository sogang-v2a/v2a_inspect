# Roadmap Overview

This roadmap is intentionally staged. The order matters because several of the current system’s weaknesses are structural, not cosmetic.

The final target is a **tool-first, agentic multitrack description pipeline** with Gemini acting as an adjudicator rather than the universal first-pass extractor.

---

## 1. Why the roadmap is staged this way

The current repository already contains useful pieces:

- a client/server runtime split
- server-side SAM3/DINOv2/SigLIP2 model loading
- a working remote-analysis path
- a partial tool-evidence layer

But the current pipeline also has key structural gaps:

- the structural unit is still too tied to Gemini scene analysis
- “where to cut” is not solved
- crop-level evidence is not yet the foundation of identity
- identity and grouping semantics are still mixed
- tool results reach Gemini mostly as prompt evidence rather than direct tool calls
- the repair loop is weak

Because of those gaps, the roadmap must proceed from **contracts and foundations** toward **structure**, then toward **identity**, then toward **semantics**, then toward **agentic orchestration**, and only then toward **final routing, validation, dataset building, and evaluation**.

---

## 2. The critical path

The shortest path to the desired end-state is:

1. lock the architecture and gold set
2. stabilize the repo and tests
3. replace arbitrary windows with candidate cuts and evidence windows
4. build real crop-based source evidence
5. separate source identity from generation grouping
6. expose direct agent tool use
7. synthesize final descriptions, route TTA/VTA, validate, and review
8. export datasets and run evaluations

Skipping any of the middle steps usually causes later work to be built on unstable assumptions.

---

## 3. Stage summary table

| Stage | Title | Suggested duration | Key early outputs |
|---|---|---|---|
| 0 | Architecture Lock, Semantic Invariants, and Gold Set | 2-3 focused days | A frozen target bundle schema., A frozen tool contract for the agent. |
| 1 | Repository Stabilization and Testable Foundations | 4-6 focused days | Lazy or safe imports for top-level packages., A lightweight test path that does not require every heavy dependency at import time. |
| 2 | Candidate Cuts, Evidence Windows, and Storyboard Artifacts | 4-6 focused days | Shot-boundary proposals., Candidate cuts with typed reasons and confidence. |
| 3 | Crop-Based Source Extraction, Re-Identification, and Identity Confidence | 1-2 focused weeks | Track crops generated from masks or boxes., Stable source-track IDs and identity confidence. |
| 4 | Event Segmentation, Ambience Beds, and Generation-Group Semantics | 1 focused week | Physical source tracks., Sound event segments derived from those source tracks. |
| 5 | Direct Agentic Tool Use, Planner/Executor, and Bounded Repair Loops | 1-2 focused weeks | A planner/executor layer., Direct tool endpoints or callable adapters. |
| 6 | Canonical Description Synthesis, TTA/VTA Routing, Validation, and Human Review | 1-2 focused weeks | Canonical descriptions per generation group., Final TTA/VTA routing decisions with rationale and confidence. |
| 7 | Dataset Builder, Evaluation, Demo Hardening, and Research Packaging | 1-2 focused weeks for first pass, then ongoing | Dataset export format and batch builder., Evaluation harness and baseline runners. |

---

## 4. Recommended reading and execution order

For each stage, follow this order:

1. read `stage_X_main.md`
2. read `stage_X_detailed.md`
3. use `stage_X_checklist.md` as the exit gate
4. only then start code changes

The detailed file should be treated as the authoritative implementation note for that stage.

---

## 5. Cross-stage rules

### 5.1 Do not re-expand scope while the semantic foundation is unstable

If the ontology is still changing, do not start inventing more models, more providers, or more UI surface. First make the current semantics reliable.

### 5.2 Keep temporary adapters explicit

It is acceptable to keep legacy adapters while migrating, but it is not acceptable to let those adapters become the de facto architecture.

### 5.3 Prefer high-recall structure before fine precision

Candidate cuts, source proposals, and grouping proposals can be conservative and somewhat noisy at first as long as confidence and evidence are explicit. Later adjudication and validators can refine them.

### 5.4 Preserve inspectability at every layer

If a later stage cannot explain how it got its output, that stage is under-specified. The project’s research value depends on inspectable intermediate artifacts.

### 5.5 Treat evaluation as part of the architecture

The gold set, validators, and baseline comparisons are not “after the real work.” They are part of the real work.

---

## 6. What can be parallelized

Some work can happen in parallel, but only after the dependencies are clear.

### Good parallelization examples

- Stage 1 test and packaging cleanup can proceed in parallel with Stage 0 doc cleanup after the blueprint is locked.
- Stage 2 storyboard tooling and cut-merging logic can be developed alongside fixture generation.
- Stage 6 UI review work can begin once Stage 4 contracts are stable, even if Stage 5 is still being polished.

### Bad parallelization examples

- implementing direct agent tool use before the final contracts exist
- building dataset export before the bundle schema is stable
- optimizing routing before identity/group semantics are fixed

---

## 7. Current-to-target milestone logic

A useful way to think about progress is the following:

### Milestone A: Trust the contracts
Achieved when contracts, invariants, and the gold set are locked.

### Milestone B: Trust the repo
Achieved when lightweight tests and fake runtimes work.

### Milestone C: Trust the structure
Achieved when candidate cuts and evidence windows are explicit.

### Milestone D: Trust the visual identity
Achieved when crop-based evidence and source identities are stable.

### Milestone E: Trust the semantics
Achieved when source tracks, event segments, and generation groups are distinct and meaningful.

### Milestone F: Trust the agent
Achieved when the reasoner calls tools directly under budgets.

### Milestone G: Trust the output
Achieved when routing, validation, and human review produce exportable bundles.

### Milestone H: Trust the research story
Achieved when dataset export and evaluation baselines are reproducible.

---

## 8. Stage directories

- [Stage 0 — Architecture Lock, Semantic Invariants, and Gold Set](stages/stage_0_architecture_lock/stage_0_main.md)
- [Stage 1 — Repository Stabilization and Testable Foundations](stages/stage_1_repo_stabilization/stage_1_main.md)
- [Stage 2 — Candidate Cuts, Evidence Windows, and Storyboard Artifacts](stages/stage_2_candidate_cuts_and_evidence/stage_2_main.md)
- [Stage 3 — Crop-Based Source Extraction, Re-Identification, and Identity Confidence](stages/stage_3_source_extraction_and_identity/stage_3_main.md)
- [Stage 4 — Event Segmentation, Ambience Beds, and Generation-Group Semantics](stages/stage_4_event_schema_and_generation_groups/stage_4_main.md)
- [Stage 5 — Direct Agentic Tool Use, Planner/Executor, and Bounded Repair Loops](stages/stage_5_agentic_adjudication/stage_5_main.md)
- [Stage 6 — Canonical Description Synthesis, TTA/VTA Routing, Validation, and Human Review](stages/stage_6_description_routing_validation/stage_6_main.md)
- [Stage 7 — Dataset Builder, Evaluation, Demo Hardening, and Research Packaging](stages/stage_7_dataset_evaluation_release/stage_7_main.md)

---

## 9. How to use this roadmap with the current repo

The existing repository should be treated as a **partial migration base**, not as a template that the final architecture must imitate exactly.

Keep what is already useful:

- client/server split
- remote heavy-inference runtime
- server model bootstrap
- Streamlit review entry point
- typed tool outputs where they already exist

Change what still reflects the transitional design:

- Gemini-first scene extraction
- arbitrary fixed-window scene planning
- whole-scene evidence for embeddings/labels
- overloaded grouping semantics
- hint-only tool usage
- weak repair loops

---

## 10. Final reminder

The roadmap is not asking the agent to polish the current pipeline into a slightly better version of itself. It is asking the agent to **finish the migration to the correct end-state** while using the current repository as leverage instead of baggage.
