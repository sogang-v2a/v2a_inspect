# MiG Stage Timing Report (2026-04-11)

## What was measured
This note captures the most concrete stage timing evidence gathered so far for the `sogang_gpu` A100 1g.10gb MiG slice.

## Server resource policy
For future debugging and benchmarking on the MiG host:
- treat the server as an **always-on shared resource**
- do **not** stop it after each step just because a check finished
- only clean it up when it is clear the resource will not be reused
- prefer the normal user-facing launch path (`uv run --project server v2a-inspect-server serve`)
- prefer HTTP requests for routine interaction and benchmarking

## Verified single-frame SAM3 timing
A direct single-frame SAM3 run on:
- `/data/artifacts/v2a_cat-1a5038e8-pfe53o1l/scene_0000_frame_00.jpg`

with prompts:
- `cat`
- `animal`
- `object`

returned these measured values:

- `sam3_client_load_seconds`: **107.42s**
- `inference_seconds`: **14.42s**
- `track_count`: **3**
- `strategy`: `scene_prompt_seeded`

## Verified device placement
The same run reported:
- `client_device_attr`: `cuda`
- `model_param_device`: `cuda:0`
- `model_dtype`: `torch.float16`

So the SAM3 inference path is using the GPU, not CPU-only fallback.

## Verified CUDA memory deltas
The same run reported:

Before runtime:
- free: **9554.94 MiB**
- total: **9728.0 MiB**

After SAM3 model load:
- free: **7888.94 MiB**
- allocated: **1644.63 MiB**
- reserved: **1666.0 MiB**

After single-frame inference:
- free: **7248.94 MiB**
- allocated: **1706.54 MiB**
- reserved: **2274.0 MiB**

## Current interpretation
The current bottleneck is not that SAM3 fails to use CUDA.
The bottleneck is:

> **SAM3 model load time on the 10GB MiG slice (~107s) dominates the early pipeline before later bundle-quality stages have a chance to matter.**

Single-frame inference itself is still expensive (~14.4s), but the model load is the largest stage seen so far.

## What is still missing
The following timing is **not yet cleanly isolated** in this note:
- subsequent same-process SAM3 inference after the first inference

That should be measured next using the persistent server process and HTTP-first workflow where possible.
