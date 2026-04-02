FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY server/pyproject.toml ./server/pyproject.toml
COPY server/README.md ./server/README.md
COPY server/model-manifest.json ./server/model-manifest.json
COPY server/src ./server/src

RUN pip install uv && \
    uv sync --locked --no-dev && \
    uv --project server sync --locked --no-dev

CMD ["uv", "--project", "server", "run", "v2a-inspect-server", "serve"]
