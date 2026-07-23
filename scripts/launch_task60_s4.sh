#!/bin/bash
# launch_task60_s4.sh — HHHH Stage 4 inference (用 best ckpt @ gs 1360)
# patience 触发后立即跑
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec/GRID

# Copy best ckpt to safe path (avoid Hydra '=' parser error)
SAFE_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp60/task6_hhhh_best.ckpt
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp60
cp /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task60_hhhh_s3_long/checkpoints/checkpoint_epoch=000_step=001360.ckpt $SAFE_CKPT
echo "[task60 s4] ckpt: $SAFE_CKPT"

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/logs
PID_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/scripts/pids
mkdir -p $LOG_DIR $PID_DIR

CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup python -m src.inference \
    experiment=tiger_inference_flat \
    data_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys \
    semantic_id_path=/home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp59/task59_hhhh_l4_sid_tensor.pt \
    num_hierarchies=4 \
    ckpt_path=${SAFE_CKPT} \
    ++should_skip_retry=true \
    paths.root_dir=/home/wlia0047/ar57/wenyu/GeneRec/GRID \
    id=task60_hhhh_s4 \
    > $LOG_DIR/task6_hhhh_s4_v1.log 2>&1 &

PID=$!
echo "$PID" > $PID_DIR/task60_hhhh_s4_pid
echo "[task60 HHHH s4] PID=$PID, expect ~5min"
