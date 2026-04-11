# V2A Inspect — Experiment Ledger (2026-04-10)

| clip_id | clip_type | pipeline_mode | status | elapsed_s | physical_sources | sound_events | generation_groups | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `v2a_smoke_black.mp4` | synthetic black clip | `tool_first_foundation` | completed | 141.53 | 0 | 0 | 1 | Operational smoke only; no foreground structure expected. |
| `v2a_smoke_black.mp4` | synthetic black clip | `agentic_tool_first` | completed | 173.04 | 0 | 0 | 1 | Operational smoke only; validates trace/bundle persistence. |
| `v2a_smoke_box.mp4` | synthetic moving-box clip | `agentic_tool_first` | completed | 157.19 | 0 | 0 | 1 | Synthetic motion still produced no sources; this motivated the foreground-recall recovery work. |
| `v2a_cat.mp4` | real-image loop (COCO cats image) | `tool_first_foundation` | aborted | >540 | — | — | — | After `aa65248`, a nontrivial remote run was started on `sogang_gpu` and kept the server busy for >9 minutes without returning a bundle before shutdown. Use this as the first follow-up benchmark once recall recovery is tuned further. |
| `v2a_cat.mp4` | real-image loop (COCO cats image) | `tool_first_foundation` | interrupted | 20 | — | — | — | After `94ad5fa`, a short remote probe confirmed `v2a_cat-runtime-trace.jsonl` is created early and already records `structural_overview` timing (`~0.66s`) before the run was intentionally stopped. |
| `v2a_cat.mp4` | real-image loop (COCO cats image) | `tool_first_foundation` | interrupted | >60 | — | — | — | After `af731c7`, plus local manifest-path and `ffprobe` fallback fixes, a fresh remote run no longer failed immediately on startup. It created `/data/artifacts/v2a_cat-54a749f2-4jlud_ap/`, recorded `structural_overview` in `v2a_cat-runtime-trace.jsonl`, then spent the observed window loading SAM3 on the 10GB MiG slice before shutdown. |
| `v2a_cat.mp4` | real-image loop (COCO cats image) | `tool_first_foundation` | interrupted | >145 | — | — | — | After the round-7 prompt-narrowed baseline patch, a fresh run created `/data/artifacts/v2a_cat-1a5038e8-pfe53o1l/` and recorded `structural_overview` with only **2 sampled frames**. The run still failed to complete before shutdown, so the remaining blocker appears to be first-pass SAM3 extraction cost/latency even after prompt narrowing and cheaper sampling. |

## Notes
- `v2a_cat.mp4` was created remotely from `https://images.cocodataset.org/val2017/000000039769.jpg`.
- The aborted cat-loop run is **not** counted as a completed benchmark; it is logged here only to capture the latest observed cost/behavior on a nontrivial clip.
- The latest cat-loop row also captures two deployment fixes that were required before recall tuning could continue:
  - server-manifest fallback for packaged server installs
  - `ffprobe` fallback for remote environments without a system ffprobe binary
- The newest cat-loop row captures the first run after making scene-prompt narrowing the default baseline and reducing initial sampling from 3 frames to 2.
- Future rows should include bundle/trace artifact paths once the runs complete successfully.
