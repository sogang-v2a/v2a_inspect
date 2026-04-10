# V2A Inspect — Experiment Ledger (2026-04-10)

| clip_id | clip_type | pipeline_mode | status | elapsed_s | physical_sources | sound_events | generation_groups | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `v2a_smoke_black.mp4` | synthetic black clip | `tool_first_foundation` | completed | 141.53 | 0 | 0 | 1 | Operational smoke only; no foreground structure expected. |
| `v2a_smoke_black.mp4` | synthetic black clip | `agentic_tool_first` | completed | 173.04 | 0 | 0 | 1 | Operational smoke only; validates trace/bundle persistence. |
| `v2a_smoke_box.mp4` | synthetic moving-box clip | `agentic_tool_first` | completed | 157.19 | 0 | 0 | 1 | Synthetic motion still produced no sources; this motivated the foreground-recall recovery work. |
| `v2a_cat.mp4` | real-image loop (COCO cats image) | `tool_first_foundation` | aborted | >540 | — | — | — | After `aa65248`, a nontrivial remote run was started on `sogang_gpu` and kept the server busy for >9 minutes without returning a bundle before shutdown. Use this as the first follow-up benchmark once recall recovery is tuned further. |
| `v2a_cat.mp4` | real-image loop (COCO cats image) | `tool_first_foundation` | interrupted | 20 | — | — | — | After `94ad5fa`, a short remote probe confirmed `v2a_cat-runtime-trace.jsonl` is created early and already records `structural_overview` timing (`~0.66s`) before the run was intentionally stopped. |

## Notes
- `v2a_cat.mp4` was created remotely from `https://images.cocodataset.org/val2017/000000039769.jpg`.
- The aborted cat-loop run is **not** counted as a completed benchmark; it is logged here only to capture the latest observed cost/behavior on a nontrivial clip.
- Future rows should include bundle/trace artifact paths once the runs complete successfully.
