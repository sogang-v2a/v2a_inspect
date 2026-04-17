# v2a-inspect-server

Server-side tooling package for remote visual inference only.
This package exists so client-side installs do not need to absorb heavy runtime dependencies on the GPU host.

Current target runtime:
- a single remote GPU server runtime
- university-hosted `sogang_gpu` is the default target
- `runtime_profile=full_gpu` is the default research profile for the 10GB A100 MiG slice
- HF only for weights bootstrap
- the GPU host does not perform semantic LLM work

What this package is responsible for:
- checking that an NVIDIA GPU is visible at startup
- verifying minimum VRAM requirements
- bootstrapping model weights from HF into cache
- serving the server-side runtime API
- running native in-container visual inference helpers

Current CLI:
- `v2a-inspect-server check`
- `v2a-inspect-server bootstrap`
- `v2a-inspect-server runtime-info`
- `v2a-inspect-server warmup`
- `v2a-inspect-server serve`

Current HTTP endpoints:
- `GET /healthz`
- `GET /readyz`
- `GET /runtime-info`
- `POST /warmup`
- `POST /bootstrap`
- `POST /infer/sam3-extract`
- `POST /infer/embed-crops`
- `POST /infer/score-labels`
