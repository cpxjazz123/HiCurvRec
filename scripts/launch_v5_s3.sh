#!/bin/bash
# Stage 3 TIGER training v5 — 5 道防线版 (应用 SmoothedValMetric 滑动平均 monitor)
# Usage: bash launch_v5_s3.sh <variant> <gpu>
#   variant: H_E_E_E | H_H_E_E | H_H_H_H
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /fs04/ar57/wenyu/GeneRec/GRID/.claude/worktrees/task9-decoder-only

VARIANT=$1
GPU=${2:-0}

case $VARIANT in
    H_E_E_E) SHORT=hee ;;
    H_H_E_E) SHORT=hhee ;;
    H_H_H_H) SHORT=hhhh ;;
esac

SID_PATH=/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_${SHORT}_s22/pickle/merged_predictions_tensor.pt
LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs
PID_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/pids
mkdir -p $LOG_DIR $PID_DIR

echo "[Stage3 v5 $VARIANT] GPU=$GPU SID=$SID_PATH"

CUDA_VISIBLE_DEVICES=${GPU} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -m src.train \
    experiment=tiger_train_flat \
    data_dir=/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys \
    semantic_id_path=${SID_PATH} \
    num_hierarchies=4 \
    paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID \
    seed=42 \
    trainer.max_steps=2560 \
    trainer.val_check_interval=320 \
    ++trainer.limit_val_batches=200 \
    callbacks.model_checkpoint.save_last=True \
    callbacks.model_checkpoint.monitor=val/recall@5_avg5 \
    callbacks.model_checkpoint.mode=max \
    callbacks.early_stopping.monitor=val/recall@5_avg5 \
    callbacks.early_stopping.mode=max \
    callbacks.early_stopping.patience=2 \
    id=task388v5_${VARIANT,,}_s3 \
    ++should_skip_retry=true \
    > $LOG_DIR/v5_${VARIANT,,}_s3.log 2>&1 &

PID=$!
echo "[Stage3 v5 $VARIANT] PID=$PID"
echo "$PID" > $PID_DIR/v5_${VARIANT,,}_pid