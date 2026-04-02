# v2a-inspect-server

Server-side tooling package for remote visual inference (SAM3, embeddings, label scoring).
This package exists so client-side installs do not need to absorb future heavy runtime dependencies.

Current target runtime:
- Docker container
- NVIDIA GPU host
- HF only for weights bootstrap

What this package is responsible for:
- checking that an NVIDIA GPU is visible at startup
- verifying minimum VRAM requirements
- bootstrapping model weights from HF into cache
- serving the server-side inference/runtime layer

Current CLI:
- `v2a-inspect-server check`
- `v2a-inspect-server bootstrap`
- `v2a-inspect-server runtime-info`
