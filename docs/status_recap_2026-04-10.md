# V2A Inspect — Current Recap (2026-04-10)

## Current state
The project is now a **silent-video, bundle-first, tool-first research prototype** centered on the resident remote server runtime.

The active system now:
- exposes only `tool_first_foundation` and `agentic_tool_first`
- removes legacy Gemini video upload and `tool_context`
- creates a silent analysis copy before downstream media work
- proposes sources from a merged stack of:
  - SigLIP2 ontology scoring
  - Gemini frame/storyboard hypotheses
  - motion-region proposals
- verifies those hypotheses explicitly before SAM3 extraction
- groups output events through explicit acoustic recipe signatures
- keeps agentic work focused on high-value ambiguity only

## Latest implemented improvements
- Added explicit `verify_scene_hypotheses(...)` handling into the active proposal flow.
- Added explicit acoustic recipe grouping and recipe-signature metadata.
- Expanded the ontology substantially beyond the tiny hardcoded prompt regime.
- Added Gemini failure short-circuiting so benchmarking does not stall on repeated scene-hypothesis / adjudication / description-writer failures.
- Marked stale planning docs as historical where they still describe removed architecture.

## Verified local status
Passed locally:
- `ruff check src server tests docs scripts`
- `unittest discover -s tests -v`
- `unittest discover -s server/tests -v`

Most recent local result:
- root tests: **40 passed**
- server tests: **62 passed**

## Verified post-redesign hardware evidence
Saved locally:
- `data/live_test_table_tennis_foundation_redesign/`

Completed on `sogang_gpu`:
- clip: `playing_table_tennis_same_class_abab_5s`
- mode: `tool_first_foundation`
- total recorded stage time: about **480.3s**
- physical sources: **27**
- sound events: **51**
- generation groups: **32**
- verified windows: **6**
- recipe signatures: **32**
- validation: `pass_with_warnings`

Partial saved redesign evidence also exists for an `agentic_tool_first` attempt on the same clip, but it is not yet a completed benchmark-quality comparison.

## Main blocker now
The main blocker is now **cost/benefit evidence**, not architecture cleanup:
- the redesigned stack clearly runs on the target hardware,
- but the temporal benchmark pack is expensive enough that we still need broader saved results before concluding the redesign and selective agentic layer are worth their cost.
