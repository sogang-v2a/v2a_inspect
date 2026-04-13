# V2A Inspect — Current Problem (2026-04-11)

## Short version
The repo is no longer blocked by:
- MiG runtime survival
- legacy agent crash bugs
- missing foundation completion on temporal clips
- public legacy Gemini/video-upload code paths

The current blocker is now:

> **The silent-video dynamic proposal stack exists, but we still need to prove that it improves temporal benchmark quality enough to justify the extra hypothesis work and selective agentic repairs.**

## What changed in this redesign
- `legacy_gemini` is gone from the public product surface.
- The old Gemini video-upload workflow is removed.
- The old `tool_context` compatibility branch is removed.
- The active pipeline now creates a silent analysis copy and uses that path everywhere downstream.
- Source discovery is no longer based on tiny hardcoded prompt lists alone.
- The active discovery stack is now:
  - SigLIP2 ontology scoring
  - Gemini frame/storyboard hypotheses
  - motion-region proposals

## What is confirmed now
- Only two supported pipeline modes remain:
  - `tool_first_foundation`
  - `agentic_tool_first`
- Public CLI/UI/server paths now route only through the tool-first server runtime.
- ffmpeg frame/clip extraction now uses silent-video discipline.
- Foundation and agentic still both complete on the saved sword-fighting temporal sample.
- The previous agentic cut-payload crash remains fixed.

## Current research question
The next question is no longer “can the system run?”
It is:

> **On the temporal benchmark pack, when does the new proposal stack improve source/event/group quality, and when does `agentic_tool_first` add enough value over `tool_first_foundation` to justify its extra latency?**

## Immediate next evidence to gather
- Re-run the temporal core clips with the redesigned silent-video proposal stack.
- Compare foundation vs agentic on:
  - source count correctness
  - event segmentation quality
  - grouping quality
  - routing plausibility
  - final description grounding
- Use the local `data/` benchmark workflow and persist outputs under `data/benchmarks/...`.
