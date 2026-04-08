# AGENT_READ_FIRST

This file exists because the implementation agent is likely to drift if the target is underspecified.

## Mission

Build a **tool-first, agentic pipeline** that converts **video without using audio** into a **multitrack audio description bundle** for downstream TTA/VTA generation, review, and dataset construction.

## What the final system is allowed to be

The final system is allowed to be:

- strongly server-centric for heavy inference
- tool-first rather than Gemini-first
- iterative and agentic
- conservative about ambiguity
- explicit about confidence and missing information
- optimized for stable research output rather than maximum novelty per line of code

## What the final system is not allowed to be

The final system is **not** allowed to become any of the following unless the blueprint is explicitly updated:

- a whole-video Gemini extractor that does everything in one or two prompts
- a system that uses audio from the input video
- a system that computes embeddings or labels from uncropped whole-scene frames once crop support exists
- a system that conflates physical source identity with generation grouping
- a system that silently merges low-confidence cross-scene identities
- a system that depends on local GPU execution for the main path
- a system that adds broad multi-provider abstraction work before the single-server path is solid
- a system whose agent can run unbounded loops without budgets, logs, or replay

## Absolute invariants

1. **No input audio use.**
2. **Remote heavy inference only.**
3. **Single-server runtime is the primary execution target.**
4. **Gemini remains in the system, but as an adjudicator and description synthesizer, not the universal first-pass extractor.**
5. **Three layers must stay distinct:**
   - physical source track
   - sound event segment
   - generation group
6. **Every important merge/split/routing decision must be inspectable later.**
7. **Human review must remain possible.**

## Execution discipline

- Read the blueprint before writing code.
- Work one stage at a time.
- Read the stage `main` file before the stage `detailed` file.
- Use the stage checklist as an exit gate.
- Do not start the next stage while blocking items in the current checklist remain incomplete.
- If a stage reveals a contradiction in the blueprint, stop and update the blueprint or ADR first. Do **not** silently improvise a different target architecture.
- If you need a temporary adapter, label it as an adapter and state what it will be replaced by.

## Cost and complexity discipline

Prefer this order:

1. deterministic structure
2. crop-based evidence
3. simple heuristics with explicit confidence
4. agent adjudication for ambiguity
5. only then consider extra models or more elaborate logic

Do **not** introduce an additional heavy model just because it is available. Add a new model only if the existing stack cannot reasonably satisfy the current stage’s exit criteria.

## Temporary compatibility policy

The current repository already has legacy pipeline objects and UI assumptions. Temporary compatibility adapters are allowed, but they must obey these rules:

- adapters must point from the old structure to the new structure
- adapters must be labeled temporary
- adapters must not become the new source of truth
- new logic should not be written against the legacy ontology unless unavoidable

## Logging policy

Every agent decision worth defending later must be logged:

- unresolved issue or ambiguity
- tool called
- tool inputs
- returned artifact or output references
- acceptance or rejection decision
- rationale
- confidence / uncertainty

If this is not logged, it did not really happen in a reproducible way.

## Preferred implementation order

- stabilize repo
- build structural evidence
- build crop-based source identity
- add event semantics
- add direct agent tool-calling
- add description/routing/validation
- add dataset and evaluation

Do not invert this sequence.

## Final reminder

A useful incomplete version of the correct architecture is better than a polished version of the wrong architecture.
