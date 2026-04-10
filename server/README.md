# v2a-inspect-server

Server-side tooling package for remote visual inference and Gemini orchestration.
This package exists so client-side installs do not need to absorb heavy runtime dependencies.

Current target runtime:
- a single remote GPU server runtime
- university-hosted `sogang_gpu` is the default target
- `runtime_profile=mig10_safe` is the default profile for the 10GB A100 MiG slice
- HF only for weights bootstrap
- Gemini stays in the pipeline; server-side tools provide visual evidence

What this package is responsible for:
- checking that an NVIDIA GPU is visible at startup
- verifying minimum VRAM requirements
- bootstrapping model weights from HF into cache
- serving the server-side runtime API
- running native in-container visual tool helpers
- uploading remote video inputs before analysis

Current CLI:
- `v2a-inspect-server check`
- `v2a-inspect-server bootstrap`
- `v2a-inspect-server runtime-info`
- `v2a-inspect-server serve`

Current HTTP endpoints:
- `GET /healthz`
- `GET /readyz`
- `GET /runtime-info`
- `POST /upload`
- `POST /bootstrap`
- `POST /analyze`
