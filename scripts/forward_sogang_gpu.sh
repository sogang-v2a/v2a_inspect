#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-sogang_gpu}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
REMOTE_PORT="${REMOTE_PORT:-8080}"

echo "Forwarding http://127.0.0.1:${LOCAL_PORT} -> ${SSH_HOST}:127.0.0.1:${REMOTE_PORT}"
exec ssh -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${SSH_HOST}"
