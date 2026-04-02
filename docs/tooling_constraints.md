# Tooling Constraints

This repository is being migrated away from a Gemini-for-everything pipeline toward
a tool-first visual pipeline.

## Hard constraints

- Do **not** extract or use audio from the input video.
- Do **not** use any local GPU path.
- Heavy inference must run remotely.

## Remote inference policy

- Primary remote host: **Runpod**
- Secondary/fallback host: **Hugging Face**
- Preferred vision backbone: **SAM3**
- Preferred visual embeddings: **DINOv2**
- Preferred post-extraction label scorer: **SigLIP2**

## GPU budget policy

- Minimize cost by preferring small GPUs.
- Default target: **RTX A4000 16GB**
- Escalate only when needed: **RTX A4500**
- Never plan or request a GPU above **24GB VRAM**

## Agent-tool policy

- Coarse domain tools should be the primary agent interface.
- Mid-level CV/provider tools may be used for recovery, debugging, or low-confidence splits.
- Text-conditioned extraction is recovery-only; default extraction should be prompt-free and label-after-extraction.
