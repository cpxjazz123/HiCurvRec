#!/bin/bash
# Issue #64 单卡 fallback launcher — GPU 全被 bloger run_gr_rec_blo.py 占用时的紧急方案 (R7).
# 单卡 16s/epoch × 200 epoch ≈ 53min. PRODUCT_DIR 仍走 /tmp (lustre 不稳教训).
set -uo pipefail

GENREC_ENV=/home/wlia0047/ar57_scratch/wenyu/genrec_env
PYTHON=${GENREC_ENV}/bin/python3

if [ ! -x "${PYTHON}" ]; then
    echo "FATAL: genrec_env missing at ${GENREC_ENV}" >&2
    exit 1
fi

PRODUCT_DIR=/tmp/v64_hab
mkdir -p "${PRODUCT_DIR}"

cd /home/wlia0047/ar57/wenyu/GeneRec

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PATH=${GENREC_ENV}/bin:${PATH}

exec ${PYTHON} -u common/stage3/stage3_train_pure_t5.py \
  --sid_npy /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/sid_output.npy \
  --product_dir ${PRODUCT_DIR} \
  --expected_sid_sha be9be8f8f3ebd4298bcabd252f42af49966e0f0d1c29428eec8ea7969874fc3e \
  --tag taskA_issue64_hab_single \
  --hyperbolic_attn_bias \
  --hab_lambda_max 0.20