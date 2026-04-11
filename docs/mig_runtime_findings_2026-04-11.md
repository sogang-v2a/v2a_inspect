# MiG Runtime Findings (2026-04-11)

## Scope
These findings summarize what was directly measured on the university `sogang_gpu` A100 `1g.10gb` MIG slice during live debugging.

## Confirmed facts
- Plain PyTorch CUDA tensors work on the server host.
- The SAM3 inference path also uses CUDA in the real server environment.
- The server should be treated as an **always-on shared resource** during debugging and benchmarking.
- The project uses **sequential inference**, not concurrent multi-model inference.

## Direct SAM3 measurement
A direct single-frame SAM3 run on:
- `/data/artifacts/v2a_cat-1a5038e8-pfe53o1l/scene_0000_frame_00.jpg`

with prompts:
- `cat`
- `animal`
- `object`

reported:
- `client_device_attr = cuda`
- `model_param_device = cuda:0`
- `model_dtype = torch.float16`
- `sam3_client_load_seconds = 107.42s`
- `inference_seconds = 14.42s`
- `track_count = 3`

## GPU memory findings
Measured from the same run:

Before runtime:
- free: `9554.94 MiB`
- total: `9728.0 MiB`

After SAM3 model load:
- free: `7888.94 MiB`
- allocated: `1644.63 MiB`
- reserved: `1666.0 MiB`

After one inference:
- free: `7248.94 MiB`
- allocated: `1706.54 MiB`
- reserved: `2274.0 MiB`

Interpretation:
- model load is the dominant cost
- one inference adds only a modest amount of true allocated memory
- PyTorch increases reserved memory noticeably after inference

## Project-aligned conclusion
For this project’s intended **sequential** pipeline, it appears fine to host:
- `sam3`
- `dinov2-base`
- `siglip2-base`

in the same always-on server process.

The main bottleneck is **not** multi-model coexistence.
The main bottleneck is:

> **SAM3 cold load / reload time on the 10GB MiG slice.**

So the optimization target should be:
- avoid repeated SAM3 cold loads
- keep the server alive and reuse the loaded process when possible
- benchmark later stages only after eliminating unnecessary reloads

## Operational rule
During debugging and benchmarking on `sogang_gpu`:
- do **not** tear down the server after each check
- only clean up when it is clear the resource will not be reused
- prefer the normal user-facing launch path
- prefer HTTP interaction for routine benchmarking
