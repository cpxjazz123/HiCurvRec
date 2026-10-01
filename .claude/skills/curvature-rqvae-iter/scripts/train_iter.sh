#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wlia0047/ar57/wenyu/GeneRec"
PY="/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

WORK="$(pwd -P)"
NAME="$(basename "$WORK")"

if [[ ! "$NAME" =~ ^curvature_RQ-VAE(_iter[0-9]+)?$ ]]; then
  echo "[train_iter] run from stage2_RQ-VAE/curvature_RQ-VAE/ or curvature_RQ-VAE_iter<N>/; got $WORK" >&2
  exit 2
fi

if [[ ! -f "$WORK/curvature_RQ-VAE.py" ]]; then
  echo "[train_iter] missing $WORK/curvature_RQ-VAE.py" >&2
  exit 2
fi

if [[ ! -f "$WORK/scripts/mvg_check.py" ]]; then
  echo "[train_iter] missing iteration-local scripts/mvg_check.py" >&2
  exit 2
fi

MVG_OUT="$("$PY" "$WORK/scripts/mvg_check.py")"
printf '%s\n' "$MVG_OUT"
if ! grep -q '^MVG PASS$' <<<"$MVG_OUT"; then
  echo "[train_iter] MVG did not emit exact 'MVG PASS'" >&2
  exit 3
fi

exec "$PY" "$WORK/curvature_RQ-VAE.py"
