#!/usr/bin/env bash
set -euo pipefail

NEXT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9}"

cd "$NEXT"
exec "$PY" scripts/grad_check.py
