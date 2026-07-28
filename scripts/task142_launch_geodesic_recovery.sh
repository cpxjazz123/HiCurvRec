#!/bin/bash
# Task #142 — Free-curv codebook collapse recovery (geodesic kmeans + dead code reset)
# 2026-07-24
#
# Tests Direction 1 (kmeans_init in geodesic space) + Direction B (dead code reset)
# of Task #137 verdict 4 unverified recovery directions.
#
# R7: GPU 1 free (0% util, 0 MiB).
# R12: best_loss auto-save per epoch.

set -uo pipefail

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task142
mkdir -p "$LOG_DIR"
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task142/_TRAINING_PID
PRODUCTS_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task142/train/arm_A_M1
mkdir -p "$PRODUCTS_DIR" "$(dirname "$PID_FILE")"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/task142_arm_A_geodesic_${TS}.log"

echo "[$(date)] Launching Task #142 Arm A: geodesic kmeans + dead code reset — log: $LOG_FILE"

nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --kappa_max=0.2 \
    --lr_theta=0.0005 \
    --geodesic_kmeans \
    --re_kmeans_every=0 \
    --dead_code_reset_every=0 \
    --theta_init=0.01 \
    --epochs=200 \
    --seed=42 \
    --ckpt_dir="$PRODUCTS_DIR" \
    --kappa_log_path="$PRODUCTS_DIR/kappa_history.json" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "[$(date)] PID=$TRAIN_PID"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
