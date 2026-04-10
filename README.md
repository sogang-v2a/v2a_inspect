# v2a_inspect

Current target runtime is:

- `v2a_inspect` client/UI
- `v2a_inspect_server` server runtime
- generic remote GPU deployment
- university-hosted `sogang_gpu` is the default target
- `runtime_profile=mig10_safe` is the default profile for the A100 10GB MiG slice
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
- Use `server/scripts/warmup_university_gpu.sh` for the university warmup path
- Use `POST /upload` followed by `POST /analyze` for remote client/server runs
