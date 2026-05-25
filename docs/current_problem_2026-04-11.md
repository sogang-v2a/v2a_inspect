# V2A Inspect — Current Problem (2026-04-11)

## Short version
The remaining blocker is no longer legacy cleanup.

The current blocker is now:

> **The bundle-first Gemini-semantic rewrite is in place, but real hardware validation is currently blocked by Gemini `RESOURCE_EXHAUSTED` failures on the configured remote project.**

## What is confirmed now
- The public surface is reduced to `tool_first_foundation` and `agentic_tool_first`.
- Legacy Gemini video-upload, `tool_context`, grouped-analysis compatibility, and old response-model adapters are gone from the active path.
- The active pipeline now includes:
  - silent analysis video ingest
  - structural cuts / windows / frames / storyboard
  - Gemini open-world source proposal
  - Gemini phrase grounding
  - SAM3 extraction
  - crop / embedding / re-ID grounding
  - Gemini source / event interpretation
  - Gemini grouping
  - Gemini routing
  - Gemini description writing
- Local validation passes after the semantic reset.

## What is not done yet
- Fresh real-hardware benchmark attempts now exist for both `tool_first_foundation` and `agentic_tool_first`, but both runs on `playing_table_tennis_same_class_abab_5s` produced empty bundles because Gemini source proposal failed with `RESOURCE_EXHAUSTED`.
- The stack now surfaces that blocker explicitly in response warnings and saved artifacts.
- We therefore still do **not** have enough evidence to judge:
  - bundle quality after the rewrite
  - cost/benefit of the re-enabled agentic layer

## Immediate next evidence to gather
- Lift or replace the remote Gemini project spend cap.
- Re-run the saved benchmark target `playing_table_tennis_same_class_abab_5s` for both modes.
- If both then complete with non-empty semantic outputs, expand to one or two additional temporal clips before updating conclusions.
