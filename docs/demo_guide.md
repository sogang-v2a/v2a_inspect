# Demo Guide

## Reproducible non-GPU demo flow

1. Run the existing analysis path to obtain a `multitrack_bundle` response.
2. Save the reviewed bundle to `review_bundle.json` through the Streamlit UI.
3. Export a dataset record:
   - use `build_dataset_record(...)`
   - write with `export_dataset_record(...)`
4. Produce baseline comparisons with `compare_baselines(...)`.
5. Capture the following screenshots or artifacts:
   - evidence windows + storyboard
   - physical sources and event segments
   - generation groups + route decisions
   - validation issues + review edits
6. Include one successful example and one failure case.

## Required artifacts

- reviewed bundle JSON
- dataset record JSON
- baseline metrics JSON/table
- one screenshot of evidence windows
- one screenshot of route override / review edit persistence

## Notes

This guide intentionally avoids GPU-specific runtime steps. GPU-backed validation
of model quality is a separate operational pass.
