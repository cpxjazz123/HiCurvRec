#!/usr/bin/env bash
set -euo pipefail

# FCCR-1 Stage2 launcher.
# Run with NO CLI arguments from:
#   stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/
#
# Current CLAUDE.md is authoritative for interpreter/path policy.

ROOT="/home/wlia0047/ar57/wenyu/GeneRec"
PY="/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

WORK="$(pwd -P)"
NAME="$(basename "$WORK")"

if [[ ! "$NAME" =~ ^curvature_RQ-VAE_iter[0-9]+$ ]]; then
  echo "[train_iter] run from stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/; got $WORK" >&2
  exit 2
fi

if [[ ! -f "$WORK/curvature_RQ-VAE.py" ]]; then
  echo "[train_iter] missing $WORK/curvature_RQ-VAE.py" >&2
  exit 2
fi

# Mandatory 2+1 deliberation gate. No Stage2 launch if any pre-Stage2 stage lacks A/B/Judge evidence.
"$PY" "$ROOT/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py"

# Mandatory contract preflight. Re-run the canonical contract check immediately before launch.
"$PY" "$ROOT/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py"

# Iteration-local MVG is mandatory and must be FCCR-1 aware.
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

mkdir -p "$WORK/logs"
nohup "$PY" "$WORK/curvature_RQ-VAE.py" > "$WORK/logs/train_run.log" 2>&1 &
echo "[train_iter] launched pid=$! work=$WORK"
echo "[train_iter] log=$WORK/logs/train_run.log"
