# V2A Inspect

V2A Inspect turns a video into an inspectable sound design workspace. It analyzes
the visual structure of a video, builds an editable sound timeline, generates
audio for the timeline, and previews the result as a mixed video.

The project has three main parts:

- the main Python app and pipeline in `src/v2a_inspect`
- the browser UI in `web`
- the GPU inference server in `server`

## What It Does

V2A Inspect helps you move from a silent or unfinished video to a structured
audio plan:

1. Normalizes the input video to a fixed working format.
2. Detects scene boundaries and extracts representative keyframes.
3. Identifies important visible objects in each scene.
4. Tracks those objects through the video with the inference server.
5. Computes visual events such as motion, appearance, disappearance, scale
   changes, and contact.
6. Builds an editable sound timeline with sources, tracks, and sound events.
7. Generates event audio, mixes track stems, and creates a preview video.

The central output is a `VideoAsset` JSON file. It contains the analyzed video
metadata, scene information, object tracks, visual events, sound timeline, and
generated audio artifact paths.

## Run The App Locally

Install the Python app, build the browser UI, then start the UI server:

```bash
uv sync --extra ui --extra observability

cd web
npm ci
npm run build

cd ..
uv run v2a ui
```

Open the app at:

```text
http://127.0.0.1:8501
```

The app needs an inference server for object tracking and video-conditioned
audio generation. By default, the client uses `http://localhost:8080`; you can
override that in the UI form or with environment variables.

## Frontend Development

For frontend work, run the Python UI backend and Vite dev server separately.

Terminal 1:

```bash
uv run v2a ui
```

Terminal 2:

```bash
cd web
npm ci
npm run dev
```

The Vite dev server proxies `/api` and `/events` to the Python UI backend on
port `8501`.

## Command Line Usage

Run the full analysis pipeline and write a `VideoAsset` file:

```bash
uv run v2a run input.mp4 \
  --server-url http://localhost:8080 \
  --output asset.json
```

Generate and mix audio from a completed `VideoAsset`:

```bash
uv run v2a synthesize \
  --video input.mp4 \
  --asset asset.json \
  --output output.mp4
```

Start the UI server with a custom host or port:

```bash
uv run v2a ui --host 0.0.0.0 --port 8501
```

## Docker Compose

The root Docker image contains the Python UI backend and a prebuilt browser UI.
It is the simplest deployment path for the main app.

GitHub Actions publishes a ready-to-run image:

```text
ghcr.io/sogang-v2a/v2a_inspect:latest
```

`docker-compose.yaml` uses that image by default. You can override it with
`V2A_INSPECT_IMAGE` if you build or publish a different image.

Create the expected secret files:

```text
secrets/gemini_api_key
secrets/langfuse_public_key
secrets/langfuse_secret_key
secrets/ui_password
```

Then start the app:

```bash
docker compose up
```

The Compose service exposes the UI on port `8501` by default and stores runtime
data in the `v2a_inspect_data` Docker volume.

## Configuration

Common app settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `V2A_INSPECT_UI_HOST` | UI bind host | `127.0.0.1` locally, `0.0.0.0` in Docker |
| `V2A_INSPECT_UI_PORT` | UI bind port | `8501` |
| `V2A_INSPECT_UI_WORK_DIR` | Uploads, work files, generated audio, and previews | system temp dir locally, `/data/work` in Docker |
| `V2A_INSPECT_UI_SERVER_URL` | Inference server URL used by the UI | empty; client defaults to localhost |
| `V2A_INSPECT_UI_PASSWORD` | Enables password login when set | unset |
| `V2A_INSPECT_CLIENT_SERVER_HOST` | Default inference server host for Python clients | `localhost` |
| `V2A_INSPECT_CLIENT_SERVER_PORT` | Default inference server port for Python clients | `8080` |
| `V2A_INSPECT_CLIENT_TIMEOUT` | HTTP timeout for inference calls | `300` |
| `V2A_INSPECT_LLM_API_KEY` | LLM key for scene analysis and timeline generation | unset |
| `GEMINI_API_KEY` | Alternate LLM key name | unset |
| `OPENAI_API_KEY` | Dialogue audio generation | unset |
| `ELEVENLABS_API_KEY` | SFX, ambience, and music generation | unset |
| `V2A_INSPECT_VIDEO_ENCODE_USE_NVENC` | Use NVIDIA NVENC when available | auto-detected locally, `true` in Docker |
| `V2A_INSPECT_LANGFUSE_PUBLIC_KEY` | Optional Langfuse tracing public key | unset |
| `V2A_INSPECT_LANGFUSE_SECRET_KEY` | Optional Langfuse tracing secret key | unset |
| `V2A_INSPECT_LANGFUSE_BASE_URL` | Langfuse server URL | `https://langfuse.riverfog7.com` |

The LLM settings also support separate small, medium, and large model names via
`V2A_INSPECT_LLM_SMALL_MODEL`, `V2A_INSPECT_LLM_MEDIUM_MODEL`, and
`V2A_INSPECT_LLM_LARGE_MODEL`.

## Project Structure

```text
.
├── src/v2a_inspect/        Main Python app, pipeline, models, UI API, and audio tools
├── web/                    React UI served by the Python app
├── server/                 GPU inference service used by the main app
├── demo/                   Example notebooks
├── scripts/                Local helper scripts
├── Dockerfile              Main app image
├── docker-compose.yaml     Main app deployment
├── pyproject.toml          Main Python package metadata
└── uv.lock                 Main Python lockfile
```

Important Python modules:

| Path | Purpose |
| --- | --- |
| `src/v2a_inspect/cli.py` | `v2a` command line entrypoint |
| `src/v2a_inspect/pipeline.py` | Full video analysis pipeline |
| `src/v2a_inspect/ui/` | FastAPI UI backend, API routes, auth, and state store |
| `src/v2a_inspect/preprocessing/` | Video normalization, scene detection, keyframes, tracking, and visual events |
| `src/v2a_inspect/models/` | `VideoAsset`, scene, tracking, visual, sound timeline, and audio artifact models |
| `src/v2a_inspect/client/` | HTTP clients for the inference server |
| `src/v2a_inspect/audio_generation/` | Audio generation and mixing |
| `src/v2a_inspect/prompts/` | Prompt templates used by the analysis pipeline |
| `src/v2a_inspect/visualization/` | Rendering helpers for timelines, tracks, masks, and notebooks |

## Development Commands

Format and lint Python code:

```bash
uv run ruff check --fix src/ server/src/
uv run ruff format src/ server/src/
```

Run the Python type checker:

```bash
uv run ty check src/
```

Build the browser UI:

```bash
cd web
npm ci
npm run build
```

Build the main Docker image:

```bash
docker build -t v2a-inspect .
```

The published main app image is:

```text
ghcr.io/sogang-v2a/v2a_inspect:latest
```

## Server

The main app calls the GPU inference server for object tracking, embeddings,
label scoring, and video-conditioned audio generation. See
[`server/README.md`](server/README.md) for setup and API details.

The published server image is:

```text
ghcr.io/sogang-v2a/v2a_inspect_server:latest
```
