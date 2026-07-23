#!/bin/bash
# task3_group_c_s2.sh — Group C Stage 2.2: SID inference (用 GSRQ 训练出的 ckpt)
# 依赖: task13_group_c_s21 完成且产出 checkpoint_000_003000.ckpt
set -euo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys

GSRQ_CKPT_DIR=/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task13_group_c_s21/checkpoints
GSRQ_CKPT=$(ls -1 "$GSRQ_CKPT_DIR"/checkpoint_*.ckpt 2>/dev/null | head -1)
if [ -z "$GSRQ_CKPT" ] || [ ! -f "$GSRQ_CKPT" ]; then
    echo "ERROR: GSRQ Stage 2.1 ckpt not found in $GSRQ_CKPT_DIR" >&2
    exit 1
fi
echo "[info] GSRQ Stage 2.1 ckpt: $GSRQ_CKPT"

CUDA_VISIBLE_DEVICES=${1:-3} python -m src.inference \
    experiment=rkmeans_inference_gsrq \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    ckpt_path=${GSRQ_CKPT} \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    paths.output_dir=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs \
    id=task13_group_c_s2 \
    ++should_skip_retry=true \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task3_group_c_s2.log
