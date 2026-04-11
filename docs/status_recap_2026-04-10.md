# V2A Inspect — Current Recap (2026-04-10)

## Current state
The project is now a **bundle-first, tool-first research prototype** running against the university GPU target (`sogang_gpu`) rather than Runpod.

As of the resident-server hot-run rerun:
- `tool_first_foundation` completes on `v2a_cat.mp4` in about **59.6s**
- `agentic_tool_first` completes on the same clip in about **108.3s**
- both runs produce non-empty structure on the university MiG host

The active system can:
- probe silent video
- propose candidate cuts / evidence windows
- sample frames and build storyboard artifacts
- run SAM3-based extraction on sampled frames
- generate crop artifacts
- run crop-backed embeddings and label scoring
- build provisional physical sources, event segments, ambience beds, and generation groups
- synthesize canonical descriptions on top of the structured bundle
- validate the bundle
- persist bundle / trace / review artifacts
- expose both `tool_first_foundation` and `agentic_tool_first`

## Important recent improvements
### User-visible / workflow
- UI and CLI now expose the real pipeline mode.
- `agentic_tool_first` is now the visible research default.
- The UI is more bundle-first and no longer depends on stale grouped-state gating.

### Description quality protection
- Generation groups now track:
  - `description_origin`
  - `description_stale`
- Review edits no longer automatically clobber writer-generated descriptions with heuristic rewrites.
- Writer-quality descriptions are preserved through review normalization unless they genuinely become stale.

### Semantics quality
- Track semantics now use **track-local** evidence rather than only window-global signals.
- Routing and event typing now consider trajectory/continuity-style features derived from actual linked observations.

### Agentic behavior
- The repair loop now has real typed ambiguity handling for:
  - cut ambiguity
  - grouping ambiguity
  - routing ambiguity
  - stale descriptions
- A Gemini-backed adjudication layer exists for ambiguous cases, on top of the bounded tool loop.
- The loop can now choose whether to accept the current bundle or run a targeted repair action.
- The agentic loop now uses **interim bundles** during repair and does a single final writer-backed bundle build after structure is accepted.
- Foundation mode no longer pays for a hidden post-hoc mutating review pass.

### Remote runtime
- Remote model loading is confirmed to use CUDA on `sogang_gpu`.
- The packaged server path now falls back to an on-repo model manifest instead of resolving only to a broken venv-local path.
- Media tooling now falls back cleanly when the remote host lacks a system `ffprobe`, which removed another immediate `/analyze` failure on `sogang_gpu`.

## Verified remote GPU facts
From `sogang_gpu`:
- `torch.cuda.is_available()` → `True`
- CUDA device name → `NVIDIA A100 80GB PCIe MIG 1g.10gb`
- server GPU check passes for the 10GB MiG slice

Measured VRAM usage during direct remote model loads:
- before runtime: ~94 MiB used
- after SAM3 load: ~1760 MiB used
- after embedding load: ~1920 MiB used
- after label load: ~1014 MiB used

So the models are genuinely loading onto the GPU.

## Verified local test status
Passed locally:
- `ruff check src tests server`
- full `unittest discover` for `tests/`
- full `unittest discover` for `server/tests/`

Most recent full local result:
- root tests: **35 passed**
- server tests: **55 passed**

## Verified remote runtime status
Validated on `sogang_gpu`:
- `/healthz` works
- `/readyz` works
- `POST /warmup` works and keeps `sam3`, `embedding`, and `label` resident
- `/analyze` works for both:
  - `tool_first_foundation`
  - `agentic_tool_first`

Recent remote smoke outputs persisted bundle artifacts successfully, e.g.:
- black clip, foundation:
  - `/data/artifacts/v2a_smoke_black-b4a5fff7-cwj5ktxw/bundle.json`
- black clip, agentic:
  - `/data/artifacts/v2a_smoke_black-1b288774-9wkzj35m/bundle.json`
- moving-box clip, agentic:
  - `/data/artifacts/v2a_smoke_box-e75100aa-tm140mk1/bundle.json`

Recent resident hot-run outputs also persisted on the cat control clip, e.g.:
- cat clip, foundation:
  - `/data/artifacts/v2a_cat-43cdb91f-nbw6ul7u/bundle.json`
- cat clip, agentic:
  - `/data/artifacts/v2a_cat-063991ad-ub038vvo/bundle.json`

## Main blocker now
The main blocker is no longer “can the MiG runtime survive?” or “can any nontrivial run complete?”.
The main blocker is now **agentic cost/benefit plus better temporal evaluation**, not runtime plumbing.

The latest honest hot-run comparison shows:
- foundation is materially faster than agentic on the current cat control clip
- both modes currently produce the same top-line structure on that clip
- the newly instrumented stage history explains more of the wall time, especially:
  - final description synthesis
  - adjudication
  - bundle persistence
- the remaining evaluation gap is that the benchmark set still leans too heavily on controls / static-image loops

At the same time:
- `v2a_smoke_box.mp4` remains a useful failure/control case
- Gemini quota exhaustion currently forces heuristic final descriptions and null adjudication decisions on the university server, so the latest benchmark is best read as a **structural** comparison rather than a full writer/judge quality comparison

That means the remaining problem is not “is the server running?” or “is CUDA real?”.
More specifically, it is:
- making the agentic layer selective enough to justify itself
- evaluating on genuinely temporal short clips from the frozen gold-set categories
- restoring writer/judge-backed quality comparisons once Gemini budget is available again

## Roadmap honesty update
The Stage 7 checklist was partially reopened because the repo has:
- export/evaluation scaffolding

but does **not yet** have all claimed completed artifacts for:
- saved baseline runs
- crop ablations
- qualitative examples / failure collections
- report-ready tables/figures

## Best next steps
1. Compare `tool_first_foundation` vs `agentic_tool_first` on a small fixed temporal clip pack, not only smoke/static controls.
2. Keep the agentic loop focused on high-value structural issues; do not spend time where it cannot plausibly change the bundle.
3. Re-run the same comparison once Gemini writer/judge budget is available so final description/adjudication quality can be measured honestly.
4. Track all successful hot runs in `docs/experiment_ledger_2026-04-10.md`.
5. Only then decide whether the next bottleneck is extraction, grouping, routing, or description quality.

## Latest implementation note
The newest recovery slice (`aa65248`) added:
- explicit foreground-collapse / missing-source repair issues
- denser window resampling
- scene-prompt foreground recovery
- more honest extraction and pipeline metadata

The newest recovery-correctness slice (`af731c7`) added:
- deterministic zero-track recovery escalation
- explicit ambience-only terminal acceptance
- richer adjudicator context
- no bogus crop repair on empty track sets

The newest remote-runtime unblockers (`0fde19f`) added:
- packaged-server manifest fallback resolution
- `ffprobe` fallback through the Python-side media stack when the remote host lacks a system binary

The newest recall-bias slice (uncommitted after `0fde19f`) added:
- prompt-narrowed default extraction instead of a brute-force default SAM3 prompt sweep
- cheaper first-pass sampling (`2` frames/window) and cheaper densification (`4` frames/window)
- more permissive singleton-track retention for hard clips

The newest observability slice (`94ad5fa`) added:
- per-stage timing history in `pipeline_metadata`
- structured recovery-attempt history
- a per-run runtime trace file written before bundle completion

Verified on `sogang_gpu`:
- a nontrivial interrupted `v2a_cat.mp4` run now creates `v2a_cat-runtime-trace.jsonl`
- the trace already records `structural_overview` timing before the run finishes, which makes slow remote runs diagnosable instead of opaque
- after the manifest-path and `ffprobe` fixes, the next fresh cat-loop run no longer failed immediately; it progressed into long SAM3 loading on the MiG slice before manual shutdown
- after the prompt-narrowed baseline patch, the next fresh cat-loop run sampled only **2 frames** in `structural_overview`, but still failed to complete before manual shutdown; this suggests the current bottleneck is still the first SAM3 extraction/load path rather than later grouping or description stages

That means the next remote benchmark should focus on whether these changes raise **foreground recall**, not just whether the server stays up.

## Commits from the latest implementation stretch
- `c8e31ab` — visible research path clearly agentic
- `0ca3bd7` — protect writer-quality descriptions through review edits
- `fa45384` — semantic decisions depend on track-local evidence
- `658629f` — repair loop judges ambiguous bundle decisions
- `f5d5aba` — media tooling runnable outside activated shell
- `737d247` — roadmap stage-7 checklist made honest again
