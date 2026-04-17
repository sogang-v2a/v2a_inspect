# Tooling Constraints

This repository now targets a **silent-video, bundle-first, Gemini-semantic pipeline**.

## Hard constraints

- Do **not** extract or use audio from the input video.
- Do **not** upload the original audio-bearing video to Gemini.
- Do **not** use any local GPU path.
- Heavy inference must run remotely.

## Remote inference policy

- Primary runtime target: **one remote GPU-hosted inference worker**
- `sogang_gpu` is the default heavy-inference target.
- Preferred vision backbone: **SAM3**
- Preferred visual embeddings: **DINOv2**
- Preferred phrase grounding scorer: **SigLIP2**
- Gemini is used for **text+image reasoning** only.
- Gemini must not receive uploaded source videos.
- The GPU host must not run semantic LLM work, grouping, routing, bundle building, or agent orchestration.

## Public pipeline policy

Only these public modes remain supported:
- `tool_first_foundation`
- `agentic_tool_first`

Removed:
- `legacy_gemini`
- the old Gemini video-upload workflow
- the old `tool_context` compatibility branch
- grouped-analysis compatibility outputs

## Architecture direction

Local orchestration handles geometry, evidence assembly, and all semantics:
1. probe / silent ingest
2. candidate cuts / evidence windows
3. sampled frames / storyboard
4. motion-region proposals
5. Gemini open-world source proposal from frames + storyboard + motion crops
6. Gemini grounding / source semantics / grouping / routing / descriptions
7. re-ID, final bundle assembly, and agentic repair

Remote GPU inference worker handles only:
1. SAM3 extraction / tracking
2. crop embedding inference
3. crop label scoring inference

## Runtime profile policy

- Default runtime profile: **`full_gpu`** for the university 10GB A100 MiG slice.
- Treat the MiG server as an **always-on resident runtime**.
- Use **`mig10_safe`** only as a fallback/debug profile.
- `cpu_dev` is for fake/unit-test execution only.
