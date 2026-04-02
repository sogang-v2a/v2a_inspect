# v2a_inspect

Current target runtime is:

- `v2a_inspect` client/UI
- `v2a_inspect_server` server runtime
- Dockerized deployment
- Runpod-hosted NVIDIA GPU runtime
- Hugging Face used only for weights bootstrap
- Gemini stays in the pipeline and consumes server-side tool evidence

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

- Publish `server/Dockerfile` to GHCR
- Run the server image on Runpod with:
  - `8080/http`
  - `SHARED_VIDEO_DIR`
  - `MODEL_CACHE_DIR`
  - `HF_TOKEN`
  - `GEMINI_API_KEY`
- Use `POST /upload` followed by `POST /analyze` for remote client/server runs
