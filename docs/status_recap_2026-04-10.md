# V2A Inspect — Current Recap (2026-04-10)

## Current state
The project is now a **silent-video, bundle-first research prototype** centered on the resident remote server runtime.

The active system now:
- exposes only `tool_first_foundation` and `agentic_tool_first`
- removes legacy Gemini video upload, `tool_context`, grouped-analysis compatibility, and legacy pipeline response models
- creates a silent analysis copy before downstream media work
- proposes sources from:
  - Gemini open-world frame/storyboard/motion-crop reasoning
  - SigLIP2 phrase grounding over Gemini-proposed phrases
  - motion-region proposals from frame differencing
- builds source/event/group/route semantics through Gemini-backed structured judgments
- keeps unresolved route/description states explicit instead of inventing fallbacks

## Latest implemented improvements
- Removed the ontology-driven proposal layer and hardcoded semantic fallbacks from the active path.
- Replaced grouped-analysis compatibility with bundle-only rendering and persistence.
- Added explicit Gemini-backed modules for:
  - source proposal
  - proposal grounding
  - source/event interpretation
  - grouping
  - routing
- Kept deterministic CV limited to geometry/evidence work.

## Verified local status
Passed locally:
- `ruff check src server tests docs scripts`
- `unittest discover -s tests -v`
- `unittest discover -s server/tests -v`

Most recent local result:
- root tests: **37 passed**
- server tests: **49 passed**

## Main blocker now
The main blocker is now **end-to-end validation of the rewritten semantics on real clips**:
- the code and tests reflect the semantic reset,
- but broader real-hardware evidence is still needed to judge bundle quality and agentic ROI.


## New observed blocker after the semantic reset
- A fresh remote benchmark on `playing_table_tennis_same_class_abab_5s` now completes on the rewritten stack, but Gemini semantic calls fail with `RESOURCE_EXHAUSTED` on the configured remote project.
- The pipeline now reports that failure explicitly in benchmark warnings instead of silently returning an apparently normal empty bundle.
- Until the Gemini project spend cap is lifted, real end-to-end semantic quality on the rewritten stack cannot be evaluated meaningfully on hardware.
