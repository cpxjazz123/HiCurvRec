#!/bin/bash
# task4_stage3_l1_brand.sh — Task #68 Stage 3 (L1 brand-augmented SID, num_hierarchies=5)
#
# 变量: SID tensor 从 (4, 11924) → (5, 11924)，新增 L1 codeword 作为第 5 hierarchy
# 保持不变: RQ-VAE ckpt, sequence_length=120, seed=42
#
# 启动命令:
#   bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task4_stage3_l1_brand.sh

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task14/from_task68/sid_with_l1.pt
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs

# 检查 SID 是否存在
if [ ! -f "${SID_PATH}" ]; then
    echo "[task68-s3] ERROR: SID not found: ${SID_PATH}" >&2
    exit 1
fi

# 检查 stage 3 resume ckpt 是否存在
RESUME_CKPT=/tmp/task5_resume_step1125.ckpt
if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[task68-s3] WARNING: resume ckpt not found, starting from scratch"
    RESUME_ARG=""
else
    RESUME_ARG="+ckpt_path=${RESUME_CKPT}"
    echo "[task68-s3] resume from: ${RESUME_CKPT}"
fi

CUDA_VISIBLE_DEVICES=1 python -m src.train \
    experiment=tiger_train_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=5 \
    sequence_length=120 \
    seed=42 \
    optim.optimizer.lr=5e-4 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=2000 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    ${RESUME_ARG} \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task4_s3.log