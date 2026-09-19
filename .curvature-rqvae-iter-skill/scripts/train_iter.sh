#!/usr/bin/env bash
# Usage: train_iter.sh <iter_id> <gpu_id>
# Example: train_iter.sh 1 0
set -euo pipefail

ITER_ID="${1:?Usage: train_iter.sh <iter_id> <gpu_id>}"
GPU_ID="${2:?Usage: train_iter.sh <iter_id> <gpu_id>}"

WORKDIR="/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter${ITER_ID}"
LOGDIR="${WORKDIR}/logs"
CKPTDIR="${WORKDIR}/out/rqvae/instruments_iter${ITER_ID}"

mkdir -p "${LOGDIR}" "${CKPTDIR}"

cd "${WORKDIR}"

# Override paths in config via env vars the script reads
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export ITER_ID="${ITER_ID}"
export CKPT_OUT_DIR="${CKPTDIR}"

echo "[train_iter] iter=${ITER_ID} gpu=${GPU_ID} ckpt=${CKPTDIR}"

# Patch MECHANISM_NAME / CKPT_OUT_DIR before launching
python3 -c "
import re, sys
cfg_path = '${WORKDIR}/curvature_config.py'
with open(cfg_path) as f:
    src = f.read()
src = re.sub(r'MECHANISM_NAME = \"[^\"]+\"', 'MECHANISM_NAME = \"iter${ITER_ID}\"', src)
src = re.sub(r'SAVE_DIR_ROOT = .*', 'SAVE_DIR_ROOT = \"${CKPTDIR}/\"', src)
with open(cfg_path, 'w') as f:
    f.write(src)
print('[config] patched MECHANISM_NAME=iter${ITER_ID}')
"

# Launch (foreground so caller can monitor; use nohup for async)
exec python3 curvature_RQ-VAE.py 2>&1 | tee "${LOGDIR}/train_iter${ITER_ID}.log"
