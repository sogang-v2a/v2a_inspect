# V2A Inspect — Current Problem (2026-04-11)

## Short version
The remaining blocker is no longer legacy cleanup or benchmark infrastructure.

The current blocker is now:

> **The redesigned silent-video proposal stack is running on real hardware, but the temporal benchmark loop is expensive enough that we still need broader saved evidence before we can claim the new stack and agentic mode are worth their cost.**

## What is confirmed now
- The public surface is reduced to `tool_first_foundation` and `agentic_tool_first`.
- Legacy Gemini video-upload and `tool_context` paths are gone.
- The active pipeline now includes:
  - silent analysis video ingest
  - ontology scoring
  - Gemini frame/storyboard hypotheses
  - explicit hypothesis verification
  - acoustic recipe grouping
- Local validation passes after the redesign and the follow-up Gemini failure short-circuiting.
- A redesigned real-hardware foundation benchmark completed on the temporal sample:
  - `playing_table_tennis_same_class_abab_5s`
  - `tool_first_foundation`
  - about **480.3s** total stage time
  - **27** physical sources / **51** sound events / **32** generation groups
  - **6** verified windows / **32** recipe signatures
  - saved locally under `data/live_test_table_tennis_foundation_redesign/`

## What is not done yet
- The full temporal core benchmark pack has **not** been completed after the redesign.
- A redesigned `agentic_tool_first` temporal run has been started, but the saved evidence is still partial rather than a clean completed side-by-side comparison.
- We therefore still do **not** have enough post-redesign evidence to answer the main research question:
  - whether the new proposal stack improves bundle quality enough to justify its extra latency
  - and whether `agentic_tool_first` adds value on temporal clips beyond the deterministic foundation path

## Immediate next evidence to gather
- Finish the post-redesign temporal core pack on `sogang_gpu` using the local forwarded workflow.
- Save completed foundation + agentic results for at least the key temporal clips.
- Update the ledger from those saved local artifacts before making architecture claims about agentic ROI.
