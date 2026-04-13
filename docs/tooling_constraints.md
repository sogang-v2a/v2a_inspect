# Tooling Constraints

This repository now targets a **silent-video, bundle-first, Gemini-semantic pipeline**.

## Hard constraints

- Do **not** extract or use audio from the input video.
- Do **not** upload the original audio-bearing video to Gemini.
- Do **not** use any local GPU path.
- Heavy inference must run remotely.

## Remote inference policy

- Primary runtime target: **one remote GPU-hosted server runtime**
- `sogang_gpu` is the default heavy-inference target.
- Preferred vision backbone: **SAM3**
- Preferred visual embeddings: **DINOv2**
- Preferred phrase grounding scorer: **SigLIP2**
- Gemini is used for **text+image reasoning** only.
- Gemini must not receive uploaded source videos.

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

Deterministic CV handles only geometry and evidence:
1. probe / silent ingest
2. candidate cuts / evidence windows
3. sampled frames / storyboard
4. motion-region proposals
5. SAM3 extraction / tracking
6. crops / embeddings / re-ID

Gemini handles semantic judgment:
1. open-world source proposal from frames + storyboard + motion crops
2. phrase grounding over Gemini-proposed phrases
3. source / event interpretation
4. generation-group merge / split judgment
5. routing judgment
6. canonical description writing

## Runtime profile policy

- Default runtime profile: **`full_gpu`** for the university 10GB A100 MiG slice.
- Treat the MiG server as an **always-on resident runtime**.
- Use **`mig10_safe`** only as a fallback/debug profile.
- `cpu_dev` is for fake/unit-test execution only.
