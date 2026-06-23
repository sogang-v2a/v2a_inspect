# V2A Inspect Server

The V2A Inspect server is the GPU inference service used by the main app. It
runs the heavy visual and audio models behind a FastAPI HTTP API, so the main
app can stay lightweight and call a remote GPU machine when needed.

The server provides:

- SAM3 object segmentation and video tracking
- DINOv2 visual embeddings
- SigLIP2 image/text label scoring
- HunyuanVideo-Foley video-conditioned audio generation

## Requirements

This package is intended for a Linux GPU environment.

Recommended runtime:

- Linux x86_64
- NVIDIA GPU
- NVIDIA container runtime for Docker, or a CUDA-capable host Python setup
- enough disk space for Hugging Face/model caches and uploaded videos

The server dependency set includes Linux CUDA wheels, so it is not expected to
install or run on macOS.

## Run With Docker

GitHub Actions publishes a ready-to-run server image:

```text
ghcr.io/sogang-v2a/v2a_inspect_server:latest
```

Run the published image with GPU access and persistent storage:

```bash
docker run --gpus all \
  -p 8080:8080 \
  -v v2a-server-data:/data \
  ghcr.io/sogang-v2a/v2a_inspect_server:latest
```

Build the server image from the repository root:

```bash
docker build -f server/Dockerfile -t v2a-inspect-server .
```

Run the locally built image:

```bash
docker run --gpus all \
  -p 8080:8080 \
  -v v2a-server-data:/data \
  v2a-inspect-server
```

The server listens on port `8080` by default.

## Run From Source

Use this path only on a compatible Linux GPU host:

```bash
cd server
uv sync --locked
uv run v2a-inspect-server serve --host 0.0.0.0 --port 8080
```

The CLI currently exposes one command:

```bash
uv run v2a-inspect-server serve
```

## Connect The Main App

Point the main app at the server:

```bash
export V2A_INSPECT_UI_SERVER_URL=http://<server-host>:8080
```

Or pass the server URL when running the command-line pipeline:

```bash
uv run v2a run input.mp4 \
  --server-url http://<server-host>:8080 \
  --output asset.json
```

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Health check |
| `POST` | `/videos/upload` | Upload a video and receive a `video_id` |
| `POST` | `/infer/sam3/track-video` | Track prompted objects through a video or frame range |
| `POST` | `/infer/sam3/segment-image` | Segment an uploaded video frame or image path |
| `POST` | `/infer/dinov2/embed-images` | Embed full images or image regions |
| `POST` | `/infer/score` | Score images against text labels with SigLIP2 |
| `POST` | `/infer/hunyuan/generate-v2a` | Generate Foley audio for a video frame range |

## Models

| Model | Used For |
| --- | --- |
| SAM3 | Text, point, or box prompted segmentation and video object tracking |
| DINOv2 | Image and region embeddings |
| SigLIP2 | Comparing image crops with text labels |
| HunyuanVideo-Foley | Video-conditioned sound effect generation |

The model clients are initialized when the FastAPI app starts. First startup can
take longer while weights are downloaded or loaded into GPU memory.

## Configuration

Common server settings:

| Variable | Purpose | Default |
| --- | --- | --- |
| `V2A_SERVER_HOST` | Server bind host | `0.0.0.0` in Docker |
| `V2A_SERVER_PORT` | Server bind port | `8080` |
| `V2A_SERVER_UPLOAD_DIR` | Uploaded videos and generated audio | `/data/uploads` in Docker, `/tmp/v2a_uploads` in source defaults |
| `V2A_SERVER_SAM31_MAX_NUM_OBJECTS` | Maximum SAM3 tracked objects | `48` |
| `V2A_SERVER_SAM31_USE_FA3` | Enable SAM3 FA3 option | `false` |
| `V2A_SERVER_SAM31_USE_ROPE_REAL` | Enable SAM3 rope-real option | `true` |
| `V2A_SERVER_SAM31_COMPILE` | Compile SAM3 model path | `false` |
| `V2A_SERVER_SAM31_WARM_UP` | Warm up SAM3 at startup | `false` |
| `V2A_SERVER_EMBEDDING_MODEL_ID` | DINOv2 model id | `facebook/dinov2-base` |
| `V2A_SERVER_LABEL_MODEL_ID` | SigLIP2 model id | `google/siglip2-base-patch16-224` |
| `V2A_SERVER_HUNYUAN_MODEL_ID` | Hunyuan model id | `tencent/HunyuanVideo-Foley` |
| `V2A_SERVER_HUNYUAN_MODEL_SIZE` | Hunyuan model size | `xl` |
| `V2A_SERVER_HUNYUAN_ENABLE_OFFLOAD` | Enable Hunyuan CPU/GPU offload | `false` |
| `V2A_SERVER_OPENCV_VIDEO_BACKEND` | OpenCV video backend | `ffmpeg` |
| `V2A_SERVER_OPENCV_FFMPEG_CAPTURE_OPTIONS` | OpenCV FFmpeg capture options | `hw_decoders_any;cuda` |
| `V2A_SERVER_ENABLE_NVENC` | Enable NVIDIA video encoding path | `true` |

Docker also sets:

| Variable | Purpose |
| --- | --- |
| `HF_HOME` | Hugging Face cache root |
| `TRANSFORMERS_CACHE` | Transformers cache root |
| `NVIDIA_VISIBLE_DEVICES` | GPU visibility for the container |
| `NVIDIA_DRIVER_CAPABILITIES` | NVIDIA runtime capabilities |

## Project Structure

```text
server/
├── pyproject.toml                 Server package metadata and dependencies
├── uv.lock                        Server lockfile
├── Dockerfile                     GPU server image
└── src/v2a_inspect_server/
    ├── app.py                     FastAPI app and HTTP routes
    ├── runtime.py                 `v2a-inspect-server` CLI
    ├── settings.py                Server configuration
    ├── inference/                 Model wrappers
    │   ├── sam3.py                SAM3 segmentation and tracking
    │   ├── embed.py               DINOv2 embeddings
    │   ├── score.py               SigLIP2 label scoring
    │   └── hunyuan.py             HunyuanVideo-Foley generation
    └── models/                    Request and response schemas
```

## Health Check

After startup, check that the HTTP server is responding:

```bash
curl http://localhost:8080/healthz
```

Expected response:

```json
{"status":"ok"}
```
