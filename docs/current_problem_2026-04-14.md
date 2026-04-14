# Current Problem (2026-04-14)

## Executive summary
The pipeline is now operational end-to-end, but it is still failing on real temporal clips for a semantic reason:

> the proposal -> grounding -> extraction-seed path is not producing usable prompts for SAM3.

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

### 2. Source proposal is too loose
The proposer can emit broad or numerous hypotheses, but they are not narrowed into a small set of concrete extractable objects.

### 3. Grounding is too conservative
Observed behavior on the table-tennis clip:
- non-zero scene hypotheses
- non-zero verified windows
- zero verified extraction prompts
- many unresolved phrases

This means the verifier is not promoting the best visible candidates into SAM3 seeds.

### 4. Empty-prompt recovery is too weak
Once verified prompts are empty, extraction collapses to zero tracks. The current pipeline does not yet have a strong fallback that turns motion-region evidence into extraction-ready seeds.

### 5. Diagnostics are still insufficient
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
2. Tighten the proposer toward 1-3 concrete extraction seeds per window.
3. Relax or redesign grounding so top concrete candidates can be promoted.
4. Add a motion-region-to-seed fallback when verifier output is empty.
5. Persist richer per-window failure diagnostics.
