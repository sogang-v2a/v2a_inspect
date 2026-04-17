# v2a_inspect

Current target runtime is:

- `v2a_inspect` client/UI
- local orchestration + semantic runtime on the client machine
- `v2a_inspect_server` remote inference runtime
- generic remote GPU deployment
- university-hosted `sogang_gpu` is the default target
- `runtime_profile=full_gpu` is the default research profile for the A100 10GB MiG slice
- Hugging Face used only for weights bootstrap
- Gemini stays in the pipeline as a frame/storyboard hypothesis model, adjudicator, and description writer

## Roadmap and architecture docs

- [Roadmap overview](docs/v2a_agent_plan/02_roadmap_overview.md)
- [Final project blueprint](docs/v2a_agent_plan/01_final_project_blueprint.md)
- [Agent tool contract](docs/agent_tool_contract.md)
- [ADR 001: target multitrack architecture](docs/adr/ADR_001_target_multitrack_architecture.md)
- [Runtime environment boundaries](docs/runtime_env.md)
- [Demo guide](docs/demo_guide.md)


## Run locally with Docker Compose

```bash
docker compose up --build
```

This starts:

- Streamlit client on `http://localhost:8501`
- inference runtime on `http://localhost:8080`

The client owns orchestration and bundle creation. The remote server is used for
visual inference RPCs only.

## Remote deployment shape

- Run the server on a remote GPU host as an inference-only worker
- `sogang_gpu` is the primary deployment target today
- Use `server/scripts/warmup_university_gpu.sh` or `POST /warmup` for the university warmup path
- Remote inference endpoints:
  - `POST /infer/sam3-extract`
  - `POST /infer/embed-crops`
  - `POST /infer/score-labels`
- Removed from the GPU host:
  - `POST /upload`
  - `POST /analyze`

## Local benchmarking against the remote university server

Keep the remote `sogang_gpu` server running, then drive experiments from this machine:

```bash
./scripts/forward_sogang_gpu.sh
```

This forwards:

- local `http://127.0.0.1:18080`
- to remote `http://127.0.0.1:8080`

The local sample pack lives in `data/video_samples.zip` and is treated as local-only scratch data.
Run the benchmark pack like this:

```bash
uv run python scripts/run_video_samples_benchmark.py
```

This will:

- extract `data/video_samples.zip` into `data/video_samples/`
- warm the forwarded server once
- run the supported `tool_first_foundation` and `agentic_tool_first` modes locally while using the forwarded GPU host only for visual inference
- save local outputs under `data/benchmarks/<run_id>/`

Tracked benchmark metadata for the sample pack lives in:

- `docs/benchmark_video_samples_manifest.json`

## Optional semantic debug harness

When Gemini quota is unavailable, the semantic LLM path can be redirected to an
OpenAI-compatible endpoint via LangChain `ChatOpenAI` for debugging only:

```bash
uv sync --extra semantic
export V2A_LLM_BASE_URL=http://127.0.0.1:8080/v1
export V2A_LLM_MODEL=gpt-5.4
```

This keeps the public runtime contract unchanged; it is an opt-in local/debug
override, not the default production backend.
