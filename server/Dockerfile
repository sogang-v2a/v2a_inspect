# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/server/.venv/bin:/app/.venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /app


FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY server/pyproject.toml ./server/pyproject.toml
COPY server/uv.lock ./server/uv.lock
COPY server/README.md ./server/README.md

RUN uv sync --locked --no-dev --no-install-project && \
    uv --project server sync --locked --no-dev --no-install-project

COPY src ./src
COPY server/model-manifest.json ./server/model-manifest.json
COPY server/src ./server/src

RUN uv sync --locked --no-dev && \
    uv --project server sync --locked --no-dev


FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

CMD ["v2a-inspect-server", "serve"]
