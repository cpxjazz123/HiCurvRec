#!/bin/bash
# Task 426 / 427 — 双消融 (dual ablation) Stage 3 + Stage 4
#
# A (task426): H_E_E_E with Flattened L1 (强制重尾, 模拟 baseline's heavy tail)
# B (task427): baseline with Dispersed L1 (打散 super-codes, 模拟 H_E_E_E's uniform)
#
# 两组并行运行, 各占一张 GPU.

set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /fs04/ar57/wenyu/GeneRec/GRID/.claude/worktrees/task9-decoder-only

GRID=/fs04/ar57/wenyu/GeneRec/GRID
LOG_DIR=$GRID/task_artifacts/scripts/logs
PID_DIR=$GRID/task_artifacts/scripts/pids
mkdir -p $LOG_DIR $PID_DIR

DATA_DIR=$GRID/data/amazon_data/toys
SID_FLATTEN=$GRID/task_artifacts/results/exp388v5/task426_dual_ablation/sid/sid_H_EEE_L1_flatten.pt
SID_DISPERSE=$GRID/task_artifacts/results/exp388v5/task426_dual_ablation/sid/sid_baseline_L1_disperse.pt

run_s3 () {
    local VARIANT=$1
    local SID=$2
    local GPU=$3
    local ID=$4
    echo "[Stage3 $VARIANT] GPU=$GPU SID=$SID ID=$ID"
    CUDA_VISIBLE_DEVICES=${GPU} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python -m src.train \
        experiment=tiger_train_flat \
        data_dir=${DATA_DIR} \
        semantic_id_path=${SID} \
        num_hierarchies=4 \
        paths.root_dir=${GRID} \
        seed=42 \
        trainer.max_steps=2560 \
        trainer.val_check_interval=320 \
        ++trainer.limit_val_batches=200 \
        trainer.devices=1 \
        callbacks.model_checkpoint.save_last=True \
        callbacks.model_checkpoint.monitor=val/recall@5_avg5 \
        callbacks.model_checkpoint.mode=max \
        callbacks.early_stopping.monitor=val/recall@5_avg5 \
        callbacks.early_stopping.mode=max \
        callbacks.early_stopping.patience=2 \
        id=${ID} \
        ++should_skip_retry=true \
        > $LOG_DIR/${ID}_s3.log 2>&1 &
    local PID=$!
    echo "$PID" > $PID_DIR/${ID}_pid
    echo "[Stage3 $VARIANT] PID=$PID"
}

run_s3 "task426_A_H_EEE_L1_flatten" $SID_FLATTEN 0 task426_A_s3
run_s3 "task427_B_baseline_L1_disperse" $SID_DISPERSE 1 task427_B_s3

echo "Both Stage 3 launched. PIDs:"
cat $PID_DIR/task426_A_s3_pid $PID_DIR/task427_B_s3_pid
