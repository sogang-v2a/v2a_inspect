# Tooling Constraints

This repository now targets a **silent-video tool-first pipeline**.

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
- Preferred label scorer / ontology retriever: **SigLIP2**
- Gemini is used only for **text+image reasoning** over sampled frames/storyboards and for structured adjudication/writing.
- Gemini must not receive uploaded source videos.

## Public pipeline policy

Only these public modes remain supported:
- `tool_first_foundation`
- `agentic_tool_first`

Removed:
- `legacy_gemini`
- the old Gemini video-upload workflow
- the old `tool_context` compatibility branch

## Runtime profile policy

- Default runtime profile: **`full_gpu`** for the university 10GB A100 MiG slice.
- Treat the MiG server as an **always-on resident runtime**.
- Use **`mig10_safe`** only as a fallback/debug profile.
- `cpu_dev` is for fake/unit-test execution only.

## Architecture direction

Deterministic CV stages should build structure first:
1. probe / silent ingest
2. candidate cuts / evidence windows
3. sampled frames / storyboard
4. ontology scoring + Gemini frame hypotheses + motion proposals
5. SAM3 extraction / tracking
6. crops / embeddings / labels
7. source / event / ambience / grouping semantics

Gemini should remain in the system, but only as:
- scene/source hypothesis proposer from image evidence
- ambiguity adjudicator
- final structured description writer
