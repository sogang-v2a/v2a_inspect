# V2A Inspect — Current Challenges for External Review (2026-04-10)

This note is for an external reviewer who wants to evaluate what is still blocking the project from the intended research outcome.

## Current project state
The repository is now a **bundle-first, tool-first research prototype** with:
- a real university GPU path (`sogang_gpu`)
- crop-backed extraction / embeddings / labels
- provisional source / event / ambience / generation-group semantics
- a bounded agent/tool loop
- a Gemini-backed adjudicator for ambiguous cases
- a description-writer stage
- persisted bundle / trace / review artifacts

The main remaining problems are no longer packaging or deployment. They are now mostly about **foreground/source recall, recovery quality, and real-clip evaluation quality**.

## Most important challenges

### 1. Foreground/source recall is still too weak on nontrivial clips
The dominant failure mode is still:
- zero physical sources
- zero sound events
- ambience-heavy or ambience-only bundles

This is the biggest blocker because every later stage depends on having usable foreground structure.

What has improved:
- the system now treats source-collapse as a first-class repair problem
- it can densify sampling and run scene-prompt recovery
- it now records recovery attempts and stage timing

What is still unresolved:
- we do not yet have a completed nontrivial remote run that proves these repairs materially improve source/event recall
- the first nontrivial remote cat-loop run still did not complete to a final bundle within the observed window

### 2. Recovery quality is improved, but not yet validated on real clips
Recovery is now materially different from baseline extraction:
- denser window sampling
- scene-specific prompt recovery
- lower-threshold recovery extraction

But the remaining question is not whether the code path exists; it is whether it actually improves results on real short clips.

Key open question:
- does the recovery ladder increase physical-source and sound-event counts on real clips enough to justify the extra MiG cost?

Current implementation note:
- the zero-track path now uses a deterministic bounded ladder (`densify_window_sampling -> recover_foreground_sources -> recover_with_text_prompt`) with explicit terminal outcomes, but this still needs real-clip evidence showing that later rungs actually improve recall instead of only increasing latency.

### 3. Nontrivial university-GPU runs are still slow and incomplete
Operationally, the server is working and GPU usage is real.

What is verified:
- `/healthz` works
- the server loads models onto CUDA
- runtime trace files now appear early in nontrivial runs
- interrupted nontrivial runs now leave stage timing evidence behind

What is still missing:
- a completed nontrivial remote bundle with convincing foreground structure
- a latency/recall tradeoff understanding for the new recovery steps

### 4. Grouping is still downstream-limited by recall
The ontology and grouping stack are much better than before, but grouping quality is still downstream of source recall.

Current situation:
- grouping heuristics and adjudication are useful once usable events/sources exist
- if the system still produces no sources, grouping quality is irrelevant because there is nothing meaningful to group

So the immediate question for external review should be:
- once recall improves, is final generation grouping using the best evidence available?

### 5. Description quality is no longer the primary blocker
The repo now has:
- a real description writer
- description provenance (`heuristic`, `writer`, `manual`)
- stale-description handling
- review logic that preserves writer-generated descriptions better

This means the next review should not focus primarily on prompt wording.

The main review question now is:
- are descriptions still being asked to rescue weak structure, or are they being written on top of sufficiently strong structure?

### 6. Validation semantics are better, but need real examples
The validator now distinguishes between:
- `accepted_ambience_only`
- `recovery_exhausted`

This is a good improvement because it separates benign ambience-only outputs from failed foreground recovery.

Important constraint:
- `accepted_ambience_only` is now intended to require an explicit terminal acceptance from the agent/human path rather than merely “some recovery happened”; this still needs validation against real ambiguous clips.

What still needs review:
- are these terminal states being assigned correctly on real clips?
- is the system stopping in the right places versus retrying too long or too little?

## What an external reviewer should focus on next
A useful external review at this stage should focus on these questions:

1. **Recall and recovery**
   - On real short clips, does the new recovery ladder materially improve foreground/source recall?
   - Are there obvious missing recovery actions or badly ordered ones?

2. **Latency vs quality on `sogang_gpu`**
   - Where is time actually going in nontrivial runs?
   - Are the new recovery actions worth their cost on the 10GB MiG slice?

3. **Cut / structure usefulness**
   - Are evidence windows and candidate cuts now good enough to support extraction, or are they still too coarse for recall-sensitive scenes?

4. **Post-recall grouping quality**
   - Once usable structure appears, does the grouping logic use identity, time, route, and event evidence well enough?

5. **Terminal validation states**
   - Does the system appropriately distinguish acceptable ambience-only cases from failed foreground recovery?

## Current evidence available in-repo
Useful files for review:
- `docs/status_recap_2026-04-10.md`
- `docs/experiment_ledger_2026-04-10.md`
- runtime traces under remote artifact directories such as:
  - `/data/artifacts/v2a_cat-*/v2a_cat-runtime-trace.jsonl`

## Bottom line
The repo has crossed the architecture/plumbing hurdle.

The main challenge is now:
> **making nontrivial clips produce usable foreground/source structure quickly enough on the university GPU, and proving that the new recovery path actually helps.**

That is the highest-value area for external review right now.
