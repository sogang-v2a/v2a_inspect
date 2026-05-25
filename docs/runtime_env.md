# Runtime environment boundaries

This project intentionally separates lightweight client/UI settings from
server-only runtime settings so that planning, contracts, and unit tests do not
need heavy runtime dependencies.

## Client-safe runtime variables

- `SERVER_BASE_URL`
- `SHARED_VIDEO_DIR`
- `REMOTE_TIMEOUT_SECONDS`
- `UI_ANALYSIS_CONCURRENCY_LIMIT`
- `UI_ANALYSIS_ACQUIRE_TIMEOUT_SECONDS`
- `UI_TEMP_CLEANUP_MAX_AGE_SECONDS`
- `UI_CLEANUP_INTERVAL_SECONDS`

These are safe for UI, orchestration, fixture, and validator code.

## Server-only runtime variables

- `HF_TOKEN`
- `MODEL_CACHE_DIR`
- `WEIGHTS_MANIFEST_PATH`
- `RUNTIME_PROFILE`
- `REMOTE_GPU_TARGET`
- `MINIMUM_GPU_VRAM_GB`
- `SERVER_BIND_HOST`
- `SERVER_BIND_PORT`
- `SHARED_VIDEO_DIR`

These should only be touched by explicit runtime code in the remote server
package.

## Heavy integration coverage that still requires the real server

- remote `/bootstrap` against the real model manifest
- remote `/healthz` and `/readyz` after model download
- remote `/upload` + `/analyze` against the deployed GPU-backed server
- model-quality validation for SAM3/DINOv2/SigLIP2 on real clips

## Lightweight non-GPU coverage

- package-root import smoke tests
- fake tooling runtime tests
- server request/response smoke tests with fake runtime patches
- contract and gold-set manifest tests
