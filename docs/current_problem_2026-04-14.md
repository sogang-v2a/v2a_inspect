# Current Problem (2026-04-14)

## Executive summary
The pipeline is now operational end-to-end, but it is still failing on real temporal clips for a semantic reason:

> the proposal -> grounding -> extraction-seed path is not producing enough **region-grounded source cards** to drive SAM3 reliably.

As a result, the system can complete a run while still returning:
- 0 extracted tracks
- 0 physical sources
- 0 sound events
- 0 generation groups

## What is working
- The silent-video pipeline runs end-to-end.
- The server is stable.
- The remote GPU path is working.
- The semantic stack can call an LLM successfully.
- Real clips reach structural stages: cuts, windows, frames, storyboard, motion regions.

## What is failing
### 1. Densification is not increasing evidence enough
The recovery path runs `densify_window_sampling`, but effective frame density is still too low to materially improve proposal quality on difficult temporal clips.

### 2. Source proposal needs region-grounded source cards
The proposer needs to tie concrete visible candidates to numbered motion-region crops / frame evidence, rather than emitting only loose phrase lists.

### 3. Grounding is too conservative
Observed behavior on the table-tennis clip:
- non-zero scene hypotheses
- non-zero verified windows
- zero verified extraction prompts
- many unresolved phrases

This means the verifier is not promoting the best visible candidates into SAM3 seeds.

### 4. Empty-prompt recovery is too weak
Once verified prompts are empty, extraction collapses to zero tracks. The pipeline needs to keep going from grounded region refs even when text prompts are sparse.

### 5. Visible-but-silent sources need an explicit state
Visible subjects that are not currently making sound should not become sound events. The semantic layer needs an explicit audibility state such as:
- `audible_active`
- `visible_but_silent`
- `background_region`
- `ambience_region`
- `unknown`

### 6. Diagnostics are still insufficient
The pipeline needs clearer persisted evidence for each failed window:
- raw proposed phrases
- grounding scores per phrase
- accept/reject rationale
- why no phrase was promoted

## Most likely bottleneck
The main bottleneck is:

> **proposal grounding / verification, not infrastructure**

The server, GPU, and execution flow are working. The semantic front-end is not yet converting visible evidence into extraction prompts reliably enough.

## Immediate next debugging targets
1. Make densification materially increase sampled evidence.
2. Tighten the proposer toward 1-3 concrete, region-grounded source cards per window.
3. Relax or redesign grounding so top concrete candidates can be promoted.
4. Add a region-seed fallback when verifier output is text-sparse.
5. Add explicit visible-but-silent handling before event creation.
6. Persist richer per-window failure diagnostics.
