#!/bin/bash
# task4_p0_euclidean_concat_s3.sh — Phase 2 P0.2 Stage 3: TIGER training
# 输入: P0.2 Stage 2.2 SID tensor
# 输出: best_tiger_*.ckpt
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task58_p0_euclidean_concat_s22/pickle/merged_predictions_tensor.pt
if [ ! -f "$SID_PATH" ]; then
    echo "ERROR: SID not found at $SID_PATH"
    exit 1
fi
echo "[task58 P0.2 s3] SID: $SID_PATH"

CUDA_VISIBLE_DEVICES=0 python -m src.train \
    experiment=tiger_train_flat \
    data_dir=${DATA_DIR} \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    sequence_length=120 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=auto \
    ++should_skip_retry=true \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    id=task58_p0_euclidean_concat_s3 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs/task4_p0_euclidean_concat_s3.log

echo "[done] ckpt at /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task58_p0_euclidean_concat_s3/checkpoints/"