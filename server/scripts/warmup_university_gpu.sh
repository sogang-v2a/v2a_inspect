#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$HOME/v2a_inspect}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080}"
SMOKE_VIDEO_PATH="${SMOKE_VIDEO_PATH:-}"
WARMUP_LOG="${WARMUP_LOG:-$ROOT_DIR/.cache/v2a_inspect_server/logs/warmup-smoke.json}"

cd "$ROOT_DIR"

mkdir -p .cache/v2a_inspect_server/models .cache/v2a_inspect_server/logs /data/uploads
command -v ffmpeg >/dev/null

python - <<'PY'
import json
import subprocess

result = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
print(json.dumps({"gpu_inventory": result.stdout.strip().splitlines()}))
PY

uv run --project server v2a-inspect-server bootstrap
curl -fsS "${SERVER_URL}/healthz"
curl -fsS "${SERVER_URL}/readyz?load_models=true"

if [[ -n "${SMOKE_VIDEO_PATH}" ]]; then
  curl -fsS -X POST "${SERVER_URL}/analyze" \
    -H 'Content-Type: application/json' \
    -d "{\"video_path\": \"${SMOKE_VIDEO_PATH}\", \"options\": {\"runtime_profile\": \"mig10_safe\"}}" \
    | tee "${WARMUP_LOG}"
fi
