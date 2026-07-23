#!/bin/bash
# launch_task60_s3.sh — HHHH Stage 3 long-train (5120 step, patience=20)
# 用 task59 SID tensor + 同样的 HHHH RQ-VAE
# 验证 task59 verdict R1 假设: "Stage 3 2560 step 不够"
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp59/task59_hhhh_l4_sid_tensor.pt

# Clean prior
rm -rf /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task60_hhhh_s3_long

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs
PID_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/pids
mkdir -p $LOG_DIR $PID_DIR

echo "[task60 HHHH s3 long-train] using SID: $SID_PATH"
echo "[task60 HHHH s3] key changes vs task59: 1) max_steps 2560->5120, 2) patience 10->20"

CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -m src.train \
    experiment=tiger_train_flat \
    data_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    seed=42 \
    trainer.max_steps=5120 \
    trainer.val_check_interval=320 \
    ++trainer.limit_val_batches=200 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.monitor=val/recall@10 \
    callbacks.model_checkpoint.save_top_k=1 \
    callbacks.model_checkpoint.mode=max \
    callbacks.early_stopping.monitor=val/recall@10 \
    callbacks.early_stopping.mode=max \
    callbacks.early_stopping.patience=20 \
    id=task60_hhhh_s3_long \
    ++should_skip_retry=true \
    > $LOG_DIR/task6_hhhh_s3_long.log 2>&1 &

PID=$!
echo "$PID" > $PID_DIR/task60_hhhh_pid
echo "[task60 HHHH s3 long] PID=$PID, expect ~5h"
