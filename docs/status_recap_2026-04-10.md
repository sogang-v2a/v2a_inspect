# V2A Inspect — Current Recap (2026-04-10)

## Current state
The project is now a **bundle-first, tool-first research prototype** running against the university GPU target (`sogang_gpu`) rather than Runpod.

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

### Remote runtime
- Remote model loading is confirmed to use CUDA on `sogang_gpu`.
- Media tooling (`ffmpeg` / `ffprobe`) now resolves from the active Python environment, which fixed remote `/analyze` failures caused by missing shell PATH activation.

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
- root tests: **31 passed**
- server tests: **46 passed**

## Verified remote runtime status
Validated on `sogang_gpu`:
- `/healthz` works
- model/runtime readiness works
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

## Main blocker now
The main blocker is **foreground/source recall and meaningful recovery**, not runtime plumbing.

Even with:
- real GPU availability
- confirmed VRAM use
- successful remote `/analyze`

recent smoke runs still produced:
- `0 source tracks`
- `0 sound events`
- only `1 generation group`

That means the remaining problem is not “is the server running?” or “is CUDA real?”
More specifically, it is:
- extraction recall on nontrivial clips
- materially different recovery actions when the first pass finds no sources
- richer real-clip evaluation after those recovery actions land

## Roadmap honesty update
The Stage 7 checklist was partially reopened because the repo has:
- export/evaluation scaffolding

but does **not yet** have all claimed completed artifacts for:
- saved baseline runs
- crop ablations
- qualitative examples / failure collections
- report-ready tables/figures

## Best next steps
1. Improve extraction quality so nontrivial clips actually produce source tracks.
2. Evaluate real short clips on `sogang_gpu`, not just smoke clips.
3. Compare `tool_first_foundation` vs `agentic_tool_first` on saved artifacts.
4. Strengthen final grouping and description quality based on those results.
5. Track real experiment outcomes in `docs/experiment_ledger_2026-04-10.md`.

## Latest implementation note
The newest recovery slice (`aa65248`) added:
- explicit foreground-collapse / missing-source repair issues
- denser window resampling
- scene-prompt foreground recovery
- more honest extraction and pipeline metadata

That means the next remote benchmark should focus on whether these changes raise **foreground recall**, not just whether the server stays up.

## Commits from the latest implementation stretch
- `c8e31ab` — visible research path clearly agentic
- `0ca3bd7` — protect writer-quality descriptions through review edits
- `fa45384` — semantic decisions depend on track-local evidence
- `658629f` — repair loop judges ambiguous bundle decisions
- `f5d5aba` — media tooling runnable outside activated shell
- `737d247` — roadmap stage-7 checklist made honest again
