# V2A Inspect — Current Recap (2026-04-10)

## Current state
The project is now a **silent-video, bundle-first, tool-first research prototype** centered on the server runtime.

The active system now:
- keeps the resident `full_gpu` MiG server as the default research runtime
- exposes only two public modes:
  - `tool_first_foundation`
  - `agentic_tool_first`
- removes the public legacy Gemini/video-upload workflow
- removes the server `tool_context` compatibility branch
- builds a silent analysis copy before downstream video processing
- proposes sources through a dynamic stack instead of tiny hardcoded prompt lists

## Active perception stack
For each evidence window, the current pipeline now combines:
- SigLIP2 ontology scoring over a larger visible-source vocabulary
- Gemini scene/source hypotheses from sampled frames or storyboard evidence
- motion-region proposals from frame differencing
- scene-specific SAM3 extraction prompts derived from those merged proposals

## Agentic posture
- `tool_first_foundation` is the deterministic silent-video structural baseline.
- `agentic_tool_first` is the selective ambiguity-repair layer on top of that baseline.
- The agentic path still uses interim bundles during repair and one final writer-backed bundle build at the end.

## What is verified
Passed locally:
- `ruff check src server tests docs scripts`
- `unittest discover -s tests -v`
- `unittest discover -s server/tests -v`

Most recent local result:
- root tests: **40 passed**
- server tests: **59 passed**

## Main blocker now
The repo is no longer blocked by runtime survival or public legacy-path confusion.
The main blocker is now **quality/latency tradeoff on real temporal clips**:
- how much the new proposal stack improves structure,
- how often agentic mode changes the bundle in a worthwhile way,
- and whether the extra latency is justified on the temporal benchmark pack.
