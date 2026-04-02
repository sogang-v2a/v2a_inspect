# Tooling Constraints

This repository is being migrated away from a Gemini-for-everything pipeline toward
a tool-first visual pipeline.

## Hard constraints

- Do **not** extract or use audio from the input video.
- Do **not** use any local GPU path.
- Heavy inference must run remotely.

## Remote inference policy

- Primary runtime target: **one Runpod-hosted Docker container on a host with an NVIDIA GPU**
- Runpod is deployment infrastructure, not an application-level provider abstraction.
- Preferred vision backbone: **SAM3**
- Preferred visual embeddings: **DINOv2**
- Preferred post-extraction label scorer: **SigLIP2**
- Remote inference runtime code and future heavy runtime deps belong in the separate
  **`v2a_inspect_server`** package, not the client package.
- **Hugging Face is used only as a model/weights source**, not as an inference backend.
- Inference runs inside the server package/runtime on an NVIDIA-enabled Docker host.

## GPU budget policy

- Minimize cost by preferring small GPUs.
- Default target: **RTX A4000 16GB**
- Escalate only when needed: **RTX A4500**
- Never plan or request a GPU above **24GB VRAM**

## Agent-tool policy

- Coarse domain tools should be the primary agent interface.
- Mid-level CV tools may be used for recovery, debugging, or low-confidence splits.
- Text-conditioned extraction is recovery-only; default extraction should be prompt-free and label-after-extraction.
- The current active runtime path should prefer the single-server execution layer over any generalized provider routing.
