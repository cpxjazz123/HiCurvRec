#!/bin/bash
# task3_group_a_s22.sh — Group A Stage 2.2: SID inference
# 输入：Group A Stage 2.1 ckpt + embedding
# 输出：merged_predictions_tensor.pt (shape (11924, 3) 或 (4, 11924) 含 dedup)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /fs04/ar57/wenyu/GeneRec/GRID

EMB_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt
DATA_DIR=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
S21_CKPT=$(ls -d /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt 2>/dev/null | head -1)
if [ -z "$S21_CKPT" ] || [ ! -f "$S21_CKPT" ]; then
    echo "ERROR: Group A Stage 2.1 ckpt not found at task13_group_a_s21/"
    exit 1
fi
echo "[info] Stage 2.1 ckpt: $S21_CKPT"

CUDA_VISIBLE_DEVICES=0 python -m src.inference \
    experiment=rkmeans_inference_flat \
    data_dir=${DATA_DIR} \
    embedding_path=${EMB_PATH} \
    embedding_dim=2048 \
    num_hierarchies=3 \
    codebook_width=256 \
    model.normalize_residuals=false \
    ckpt_path=${S21_CKPT} \
    ++should_skip_retry=true \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    id=task13_group_a_s22 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task3_group_a_s22.log
