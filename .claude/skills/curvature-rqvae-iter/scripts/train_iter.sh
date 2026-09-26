#!/usr/bin/env bash
# Usage:  scripts/train_iter.sh <iter_id>
# Example: scripts/train_iter.sh 1
#
# Hard-coded (Project Rules §1 / §3 / §4 / §5):
#   - 0 argparse / 0 CLI flag (env-var ITER_ID is the only injected value)
#   - 解释器: genrec_env python3.10 (裸 python3 无 torch)
#   - 训练入口: stage2_RQ-VAE/curvature_RQ-VAE/curvature_RQ-VAE.py
#   - launcher 由脚本内置的 _launch_via_torchrun 自动 fork 4 卡 DDP
set -euo pipefail

ITER_ID="${1:?Usage: train_iter.sh <iter_id>}"
ROOT="/home/wlia0047/ar57/wenyu/GeneRec"
WORK="${ROOT}/stage2_RQ-VAE/curvature_RQ-VAE_iter${ITER_ID}"
LOG="${WORK}/logs/train_iter${ITER_ID}.log"
PYTHON="/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10"

if [ ! -d "${WORK}" ]; then
  echo "[train_iter] missing WORK=${WORK}; 先 cp -r curvature_RQ-VAE 到这里" >&2
  exit 2
fi

mkdir -p "${WORK}/logs"

echo "[train_iter] iter=${ITER_ID} work=${WORK} log=${LOG}"
echo "[train_iter] launcher: ${PYTHON} curvature_RQ-VAE.py (内置 _launch_via_torchrun fork 4 卡 DDP)"

cd "${WORK}"
ITER_ID="${ITER_ID}" nohup "${PYTHON}" curvature_RQ-VAE.py > "${LOG}" 2>&1 &
echo "[train_iter] wrapper pid=$!"
echo "[train_iter] 监控: tail -f ${LOG}  或  tail -f ${WORK}/logs/train_migrated.log"