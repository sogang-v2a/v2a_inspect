# V2A Inspect — Current Problem (2026-04-11)

## Short version
The project is no longer blocked by:
- MiG survivability
- server startup/warmup shape
- foundation-mode completion on real hardware
- the cut-ambiguity payload/signature crash in the agentic loop

The current blocker is now:

> **the full agentic path completes, but it is still too slow and is not yet showing enough value over foundation mode on the current temporal benchmark.**

## What is confirmed working now
- The resident `full_gpu` server on `sogang_gpu` is operational through the normal HTTP path.
- Local SSH forwarding works from this machine to the remote server.
- Local benchmark artifacts can be saved under `data/`.
- `tool_first_foundation` completes on the local sword-fighting temporal sample and saves a full bundle.
- `agentic_tool_first` now also completes on that same sample after fixing the executor payload filtering bug.
- The first concrete agentic crash is fixed:
  - `cut_ambiguity` diagnostic payload fields no longer crash strict tool handlers.

## Latest live benchmark evidence
### Foundation — `13_sword_fighting.mp4`
Saved under:
- `data/live_test_sword_fighting_foundation/`

Observed:
- elapsed: about **223.8s**
- physical sources: **9**
- sound events: **9**
- generation groups: **8**
- validation: `pass_with_warnings`

### Agentic — `13_sword_fighting.mp4`
Saved under:
- `data/live_test_sword_fighting_agentic/`

Observed:
- elapsed: about **503.0s**
- physical sources: **9**
- sound events: **9**
- generation groups: **8**
- validation: `pass_with_warnings`
- adjudicator call count: **3**
- agent review tool calls: **3**
- agent review issue count: **2**
- pipeline version: `agentic_tool_first`

## What is not good enough yet
The current problem is not “agentic mode crashes.”
That part is fixed.

The current problem is:

1. **Agentic mode is much slower than foundation mode** on the live sword-fighting benchmark.
   - foundation: ~224s
   - agentic: ~503s

2. **Top-line structural results are currently the same** on this benchmark.
   - both produced 9 sources / 9 events / 8 groups
   - so the extra agentic cost is not yet justified by a clear measurable structural win

3. **Final descriptions are still heuristic in the completed agentic run.**
   - the run exercised the agentic path
   - adjudication was attempted
   - but final writer-backed descriptions did not land in the saved bundle

4. **The benchmark story is still thin.**
   - we now have a real temporal sample success
   - but we still need a broader temporal clip set to determine whether agentic mode adds value on harder cases rather than just adding latency

## Immediate next question
The next question is no longer:
- “why does the agentic run 500?”

The next question is:

> **Which parts of the agentic loop are consuming the extra ~280 seconds, and on which temporal clips does that extra work actually improve the bundle enough to justify itself?**

## Practical interpretation
The project has crossed another threshold:
- foundation path works on real temporal video
- agentic path now also completes

So the repo is no longer in a “make it run at all” phase.
It is now in a:
- **cost/benefit**
- **quality-delta**
- **temporal benchmark evaluation**
phase.
