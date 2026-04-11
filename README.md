# v2a_inspect

Current target runtime is:

- `v2a_inspect` client/UI
- `v2a_inspect_server` server runtime
- generic remote GPU deployment
- university-hosted `sogang_gpu` is the default target
- `runtime_profile=full_gpu` is the default research profile for the A100 10GB MiG slice
- Hugging Face used only for weights bootstrap
- Gemini stays in the pipeline and consumes server-side tool evidence

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
- server runtime on `http://localhost:8080`

The client saves uploaded videos into the shared `/data/uploads` volume so the
server can analyze the same real files.

## Remote deployment shape

- Run the server on a remote GPU host
- `sogang_gpu` is the primary deployment target today
- Use `server/scripts/warmup_university_gpu.sh` or `POST /warmup` for the university warmup path
- Use `POST /upload` followed by `POST /analyze` for remote client/server runs

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
- run both `tool_first_foundation` and `agentic_tool_first`
- save local outputs under `data/benchmarks/<run_id>/`

Tracked benchmark metadata for the sample pack lives in:

- `docs/benchmark_video_samples_manifest.json`
