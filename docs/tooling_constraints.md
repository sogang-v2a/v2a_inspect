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

## Future architecture direction

- Once the tool layer is reliable, the long-term target should shift from a Gemini-centered pipeline with tool augmentation toward a **tool/model-first** pipeline.
- Deterministic CV stages should build structure first:
  - probe
  - scene split
  - segmentation / tracking
  - crop generation
  - embeddings / clustering / candidate groups
- **Gemini should remain in the system**, but as an **adjudicator** rather than the primary extractor.
- Gemini should be concentrated on:
  - ambiguity resolution
  - semantic consolidation / canonical naming
  - hard merge / split decisions
  - exceptional verification
  - final explanation / summarization
- Do **not** remove Gemini entirely, but do **not** keep Gemini as the first-pass extractor for every stage once tool outputs are trustworthy.
- Priority order:
  1. make tools reliable
  2. redesign architecture around trustworthy tool outputs
