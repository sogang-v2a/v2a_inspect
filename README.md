# v2a_inspect

Current target runtime is:

- `v2a_inspect` client/UI
- `v2a_inspect_server` server runtime
- Dockerized deployment
- NVIDIA GPU host
- Hugging Face used only for weights bootstrap

## Run locally with Docker Compose

```bash
docker compose up --build
```

This starts:

- Streamlit client on `http://localhost:8501`
- server runtime on `http://localhost:8080`

The client saves uploaded videos into the shared `/data/uploads` volume so the
server can analyze the same real files.
