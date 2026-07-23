#!/bin/bash
# launch_task59_s3.sh — HHHH Stage 3 + 4 launch (修复 v5 + SID shape)
# Usage: bash launch_task59_s3.sh
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

# task59: 用 task59_hhhh_l4_sid_tensor.pt (4, 11924) 替代 task57_HHHH_s22 (12288, 3) — shape mismatch fix
# v5 用 task388v4_hhhh_s22 (4, 11924) — pre-fix RQ-VAE recipe
# task59 用 task59 hhhh v3-recipe (post-task468 fix) + correct L=4 transform
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp59/task59_hhhh_l4_sid_tensor.pt

# Clean prior failed attempt
rm -rf /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task59_hhhh_s3
rm -f /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/pids/task59_hhhh_pid

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs
PID_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/pids
mkdir -p $LOG_DIR $PID_DIR

echo "[task59 HHHH s3 v3-recipe L=4] using SID: $SID_PATH"
echo "[task59 HHHH s3] key fixes vs v5: 1) v3 z-score+KMeans recipe SID, 2) L=4 shape (4, 11924), 3) early_stopping.patience=10"

CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -m src.train \
    experiment=tiger_train_flat \
    data_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    seed=42 \
    trainer.max_steps=2560 \
    trainer.val_check_interval=320 \
    ++trainer.limit_val_batches=200 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.monitor=val/recall@10 \
    callbacks.model_checkpoint.save_top_k=1 \
    callbacks.model_checkpoint.mode=max \
    callbacks.early_stopping.monitor=val/recall@10 \
    callbacks.early_stopping.mode=max \
    callbacks.early_stopping.patience=10 \
    id=task59_hhhh_s3 \
    ++should_skip_retry=true \
    > $LOG_DIR/task6_hhhh_s3_v2.log 2>&1 &

PID=$!
echo "$PID" > $PID_DIR/task59_hhhh_pid
echo "[task59 HHHH s3 v2] PID=$PID"
