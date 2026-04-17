#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/workspace/v2a_inspect}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080}"
UV_BIN="${UV_BIN:-/workspace/uv}"

cd "$ROOT_DIR"

mkdir -p .cache/v2a_inspect_server/models .cache/v2a_inspect_server/logs /data/uploads
command -v ffmpeg >/dev/null

python3 - <<'PY'
import json
import subprocess

result = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
print(json.dumps({"gpu_inventory": result.stdout.strip().splitlines()}))
PY

"${UV_BIN}" run --project server v2a-inspect-server bootstrap
curl -fsS "${SERVER_URL}/healthz"
curl -fsS "${SERVER_URL}/readyz"
curl -fsS -X POST "${SERVER_URL}/warmup"
