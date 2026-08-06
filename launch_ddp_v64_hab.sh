#!/bin/bash
# Issue #64 DDP 4 卡训练 launcher (R32 DDP 例外允许 .sh 包装).
# 直接用 genrec_env 绝对路径 (genrec_env 在 ar57_scratch, 非 conda 创建).
# 关键修复 (3 次 lustre OST hang 教训):
#   1. PRODUCT_DIR=/tmp (ext4 本地盘) — 避免 ckpt save + log write 卡 lustre OST
#   2. launcher stdout 也走 /tmp — 避免 4 worker 抢同一 lustre stdout fd
#   3. 训练完由 stage4_eval launcher 自动 cp /tmp 产物到 _history (lustre 此时已恢复)
set -uo pipefail

GENREC_ENV=/home/wlia0047/ar57_scratch/wenyu/genrec_env
PYTHON=${GENREC_ENV}/bin/python3
TORCHRUN=${GENREC_ENV}/bin/torchrun

if [ ! -x "${PYTHON}" ] || [ ! -x "${TORCHRUN}" ]; then
    echo "FATAL: genrec_env missing or incomplete at ${GENREC_ENV}" >&2
    exit 1
fi

PRODUCT_DIR=/tmp/v64_hab
mkdir -p "${PRODUCT_DIR}"

cd /home/wlia0047/ar57/wenyu/GeneRec

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export PATH=${GENREC_ENV}/bin:${PATH}

# launcher stdout → /tmp, 完全避免 lustre IO
exec ${TORCHRUN} \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  common/stage3/stage3_train_pure_t5.py \
  --sid_npy /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/sid_output.npy \
  --product_dir ${PRODUCT_DIR} \
  --expected_sid_sha be9be8f8f3ebd4298bcabd252f42af49966e0f0d1c29428eec8ea7969874fc3e \
  --tag taskA_issue64_hab_ddp \
  --hyperbolic_attn_bias \
  --hab_lambda_max 0.20