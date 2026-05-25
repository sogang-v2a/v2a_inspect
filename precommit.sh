#!/usr/bin/env bash

set -euo pipefail

uv run ruff check --fix src/ server/src/
uv run ruff format src/ server/src/
uv run ty check src/
