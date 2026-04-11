# V2A Inspect — Current Problem (2026-04-10)

## Short version
The project is no longer blocked by packaging, MiG model coexistence, or inability to complete a nontrivial run.
The current blocker is:

> **the benchmark is now structurally working on `sogang_gpu`, but the agentic layer is still expensive relative to its measured value, and the evaluation set is still too weakly temporal to justify the extra cost.**

## What is already working
- The server runs as a resident `full_gpu` runtime on `sogang_gpu`.
- `/healthz`, `/readyz`, `/runtime-info`, and `POST /warmup` all work through the normal HTTP path.
- Hot `tool_first_foundation` completes on `v2a_cat.mp4`.
- Hot `agentic_tool_first` also completes on `v2a_cat.mp4`.
- Foundation no longer runs a hidden post-hoc mutating review pass.
- Agentic mode now uses interim bundles during the repair loop and does a single final writer-backed bundle build at the end.
- Stage history now records description synthesis, adjudication, and bundle persistence timing.
- Writer/adjudicator failures now fall back instead of crashing the whole run.

## What is failing now
On the current hot-run benchmark:
- `tool_first_foundation` completed `v2a_cat.mp4` in about **59.6s**
- `agentic_tool_first` completed the same clip in about **108.3s**
- both runs produced the same top-line structure:
  - `3` physical sources
  - `3` sound events
  - `2` generation groups

So the practical failure mode has shifted:
- the agentic path is still materially slower without a demonstrated quality win on the current clip
- the benchmark set is still dominated by controls / static-image loops rather than truly temporal clips
- Gemini quota exhaustion currently forces heuristic final descriptions and null adjudication decisions, so the latest benchmark mostly measures **structural** cost/benefit rather than full writer/judge quality

## Important clarification about Gemini
The codebase still contains Gemini-backed components:
- `GeminiDescriptionWriter`
- `GeminiIssueJudge`

But the latest resident hot-run rerun hit **Gemini quota exhaustion** (`RESOURCE_EXHAUSTED: 429`) on the university server.
The pipeline now falls back cleanly:
- final descriptions remain heuristic if the writer fails
- adjudication returns `None` and the deterministic plan continues if the judge fails

So the immediate blocker is **not** pipeline survivability anymore.
The immediate blocker is:
- benchmarking honest structural agentic ROI
- then re-running the comparison on truly temporal clips once writer/judge budget is available again

## Highest-priority next step
The next work should focus on one question:

> **On a small fixed temporal clip pack, when does `agentic_tool_first` actually improve structure or grouping enough to justify its extra time over `tool_first_foundation`?**

Until that is answered, the next optimizations should focus on:
- selective agent activation for high-value issues only
- temporal benchmark coverage beyond smoke/static controls
- restoring writer/judge-backed quality evaluation once Gemini budget is available
