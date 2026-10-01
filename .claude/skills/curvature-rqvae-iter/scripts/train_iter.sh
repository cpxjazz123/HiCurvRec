#!/usr/bin/env bash
# Run the mechanism's MVG check, then Stage2, from the single in-place
# working directory. Iterations no longer create curvature_RQ-VAE_iter<N>/;
# Git is the iteration boundary (see SKILL.md 0.2/0.3).
set -euo pipefail

PY="/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

WORK="$(pwd -P)"
NAME="$(basename "$WORK")"

if [[ "$NAME" != "curvature_RQ-VAE" ]]; then
  echo "[train_iter] run from stage2_RQ-VAE/curvature_RQ-VAE/; got $WORK" >&2
  exit 2
fi

if [[ ! -f "$WORK/scripts/mvg_check.py" ]]; then
  echo "[train_iter] missing $WORK/scripts/mvg_check.py" >&2
  exit 2
fi

if [[ ! -f "$WORK/scripts/run_stage2_curvature.py" ]]; then
  echo "[train_iter] missing $WORK/scripts/run_stage2_curvature.py" >&2
  exit 2
fi

MVG_OUT="$("$PY" "$WORK/scripts/mvg_check.py")"
printf '%s\n' "$MVG_OUT"
if ! grep -q 'MVG PASS' <<<"$MVG_OUT"; then
  echo "[train_iter] MVG did not report MVG PASS" >&2
  exit 3
fi

exec "$PY" "$WORK/scripts/run_stage2_curvature.py"
