# MiG Hot-Run Benchmark (2026-04-11)

## Scope
This note records the first real resident-server benchmark run on `sogang_gpu` after switching the university runtime policy to resident `full_gpu`.

Server/runtime shape used:
- launch path: `uv run --project server v2a-inspect-server serve`
- interaction path: HTTP only
- effective runtime profile: `full_gpu`
- residency mode: `resident`
- server treated as always-on

## Warmup
`POST /warmup` on the resident server reported:
- total wall time: `115.836s`
- `sam3`: `93.1354s`
- `embedding`: `6.7887s`
- `label`: `15.6025s`

Interpretation:
- cold warmup is still dominated by SAM3 load time
- the resident-server strategy matters because this cost is paid once

## Real hot-run results

### `v2a_cat.mp4` — `tool_first_foundation` — hot run #1
- wall time: `76.275s`
- result:
  - `3` physical sources
  - `3` sound events
  - `2` generation groups
  - validation: `pass_with_warnings`
- stage timings:
  - `structural_overview`: `0.5919s`
  - `extract_entities`: `31.4149s`
  - `crop_tracks`: `0.0091s`
  - `embed_track_crops`: `6.1313s`
  - `score_track_labels`: `6.3026s`
  - `refine_candidate_cuts`: `0.0003s`
  - `build_source_semantics`: `0.0010s`
- metadata:
  - `warm_start: true`
  - resident models before run: `sam3`, `embedding`, `label`
  - resident models after run: `sam3`, `embedding`, `label`, `description_writer`

### `v2a_cat.mp4` — `tool_first_foundation` — hot run #2
- wall time: `67.052s`
- result:
  - `3` physical sources
  - `3` sound events
  - `2` generation groups
  - validation: `pass_with_warnings`
- stage timings:
  - `structural_overview`: `0.6640s`
  - `extract_entities`: `25.7344s`
  - `crop_tracks`: `0.0103s`
  - `embed_track_crops`: `11.3161s`
  - `score_track_labels`: `2.7167s`
  - `refine_candidate_cuts`: `0.0006s`
  - `build_source_semantics`: `0.0016s`
- metadata:
  - `warm_start: true`
  - same source/event/group counts as hot run #1

### `v2a_cat.mp4` — `agentic_tool_first`
- wall time: `182.036s`
- result:
  - `3` physical sources
  - `3` sound events
  - `2` generation groups
  - validation: `pass_with_warnings`
- measured stage timings:
  - `structural_overview`: `0.5750s`
  - `extract_entities`: `23.4434s`
  - `crop_tracks`: `0.0086s`
  - `embed_track_crops`: `7.3928s`
  - `score_track_labels`: `5.1023s`
  - `refine_candidate_cuts`: `0.0009s`
  - `build_source_semantics`: `0.0016s`
  - `agent:recover_with_text_prompt`: `18.0980s`
- agent metadata:
  - `recovery_actions`: `recover_with_text_prompt`
  - `agent_review_issue_count`: `3`
  - `agent_review_tool_calls`: `3`
- interpretation:
  - agentic mode completes on real hardware
  - but is currently much slower than foundation on this clip
  - there is still meaningful wall-clock time outside the currently instrumented stage list

### `v2a_smoke_box.mp4` — `tool_first_foundation`
- wall time: `22.957s`
- result:
  - `0` physical sources
  - `0` sound events
  - `1` ambience bed
  - `1` generation group
  - validation warning: `missing_dominant_source`
- stage timings:
  - `structural_overview`: `0.2313s`
  - `extract_entities`: `17.5995s`
  - later structural stages: effectively negligible
- interpretation:
  - the resident hot runtime is working
  - foreground recall still collapses on at least some clips

## Current conclusion
The current bottleneck has shifted.

It is no longer primarily:
- server startup
- model coexistence
- cold server load for every request

It is now primarily:
- hot-run extraction latency, especially `extract_entities`
- still-uninstrumented wall-clock time around later agentic steps
- foreground recall quality on harder clips

## Operational state
After these benchmarks:
- the remote server remained running
- `/runtime-info` reported:
  - `runtime_profile: full_gpu`
  - `effective_runtime_profile: full_gpu`
  - `residency_mode: resident`
  - resident models including `sam3`, `embedding`, `label`, and later Gemini-backed clients after agentic execution
