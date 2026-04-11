# V2A Inspect — Current Problem (2026-04-10)

## Short version
The project is no longer blocked by packaging or basic remote runtime setup.
The current blocker is:

> **the nontrivial visual pipeline is still too slow and too weak at foreground/source recall on `sogang_gpu`, and it often fails before reaching the later bundle-quality stages.**

## What is already working
- The server runs on the university GPU target (`sogang_gpu`).
- `/healthz` works.
- Plain `/readyz` works.
- The packaged-server manifest fallback is fixed.
- Missing-system-`ffprobe` no longer causes immediate `/analyze` failure.
- Nontrivial runs now create remote artifact directories and runtime traces.
- The agentic recovery ladder is now more correct:
  - `densify_window_sampling`
  - `recover_foreground_sources`
  - `recover_with_text_prompt`
- Ambience-only acceptance now requires explicit terminal resolution.

## What is failing now
On nontrivial remote clips such as `v2a_cat.mp4`:
- the run reaches `structural_overview`
- even after the prompt-narrowed baseline patch, the run can still spend the observed window in the first SAM3 load / extraction work on the 10GB MiG slice
- and does **not** yet complete to a useful bundle with foreground structure

So the practical failure mode is still:
- no completed nontrivial bundle
- no proven improvement in physical-source recall
- no evidence yet that the later recovery ladder steps materially improve results on real clips

## Important clarification about Gemini
The codebase does contain Gemini-backed components:
- `GeminiDescriptionWriter`
- `GeminiIssueJudge`

But in the latest remote validation there is **no evidence that Gemini was called**.
The run appears to stall before reaching:
- description writing
- adjudication

So the immediate blocker is **not** the Gemini API key.
The immediate blocker is still the **early visual stage cost + recall problem**.

## Highest-priority next step
The next work should focus on one question:

> **Why does the nontrivial MiG run still spend so long in the first SAM3 extraction even after prompt narrowing and cheaper sampling, and how do we improve foreground recall enough to produce at least one physical source and one sound event on real short clips?**

Until that is solved, later improvements to grouping, adjudication quality, and writer quality will remain downstream of weak or missing structure.
