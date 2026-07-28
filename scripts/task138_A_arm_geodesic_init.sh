#!/bin/bash
# Task #138 — A-arm geodesic kmeans init launcher (R88/R137 follow-up).
#
# R11.4 dry-run: writes fork, NOT launch until C-arm finishes.
# C-arm (Task #137 PID 2932710) is currently occupying GPU 1, ~ep 140/1000, ETA ~24 min.
# This launcher MUST wait for C-arm to die before claiming GPU 1.
#
# A-arm (M=1, κ_max=0.5, 1000 ep) trains with:
#   --geodesic_kmeans        (A方案: expmap0->kmeans->logmap0 init)
#   --re_kmeans_every 50     (A方案: re-init codebook in current manifold every 50 ep)
#   --dead_code_reset_every 0  (B方案 DISABLED: batch 100 reset triggers NaN via R137 guard)
#                              (R11.4 dry-run: A+B combo crashes; need A-only validation first)
#   --theta_init 0.01        (R137: escape Euclidean fixed point)
#
# Output: products/task138/train/arm_A_M1/best_loss_model.pth
# PID file: products/task138/_TRAINING_PID (R88 daemon watcher)

set -e

C_ARM_PID=2932710
PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task138
LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs

# Source env
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

# Pre-create directories
mkdir -p $PROD_DIR/train/arm_A_M1
mkdir -p $LOG_DIR

# R88: PID file written after successful wait + pre-launch
PID_FILE=$PROD_DIR/_TRAINING_PID

# ----------------------------------------------------------------------
# WAIT FOR C-ARM (PID 2932710) — DO NOT START A-ARM UNTIL IT DIES
# ----------------------------------------------------------------------
TS_WAIT_START=$(date +%s)
echo "[$(date)] Task #138 A-arm launcher: waiting for C-arm PID $C_ARM_PID to exit..." | tee -a $LOG_DIR/task138_launcher.log

while kill -0 $C_ARM_PID 2>/dev/null; do
    ELAPSED=$(( $(date +%s) - TS_WAIT_START ))
    if [ $ELAPSED -gt 10800 ]; then
        # 3 hours max wait (C-arm 1000 ep ~2.5h). Refuse silent wait forever (R2).
        echo "[$(date)] ERROR: C-arm still alive after ${ELAPSED}s (>3h). Aborting." | tee -a $LOG_DIR/task138_launcher.log
        exit 1
    fi
    sleep 60
done

WAIT_DURATION=$(( $(date +%s) - TS_WAIT_START ))
echo "[$(date)] C-arm (PID $C_ARM_PID) exited after ${WAIT_DURATION}s wait." | tee -a $LOG_DIR/task138_launcher.log

# Sanity: GPU 1 should be free. If not, abort (R7).
GPU1_FREE=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1 | awk -F, '{print $1" "$2}')
echo "[$(date)] GPU 1 status after C-arm exit: util/mem = $GPU1_FREE" | tee -a $LOG_DIR/task138_launcher.log
GPU1_UTIL=$(echo $GPU1_FREE | awk '{print $1}')
if [ "${GPU1_UTIL%.*}" -gt 5 ]; then
    echo "[$(date)] ERROR: GPU 1 still busy (util=${GPU1_UTIL}%) after C-arm exit. Aborting." | tee -a $LOG_DIR/task138_launcher.log
    exit 1
fi

# ----------------------------------------------------------------------
# LAUNCH A-ARM (M=1, geodesic_kmeans=ON, dead_code_reset=ON)
# ----------------------------------------------------------------------
CKPT_DIR=$PROD_DIR/train/arm_A_M1
KAPPA_LOG=$CKPT_DIR/kappa_history.json
ARM_LOG=$LOG_DIR/task138_arm_A_M1_$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/').log

echo "[$(date)] Launching Task #138 A-arm (M=1, geodesic_kmeans=ON)..." | tee -a $LOG_DIR/task138_launcher.log
echo "  ckpt_dir: $CKPT_DIR" | tee -a $LOG_DIR/task138_launcher.log
echo "  flags: --geodesic_kmeans --re_kmeans_every 50 --dead_code_reset_every 0 --theta_init 0.01" | tee -a $LOG_DIR/task138_launcher.log
echo "  R11.4: B方案禁用 (dead_code_reset 触发 NaN, 先验证 A 方案独立健康)" | tee -a $LOG_DIR/task138_launcher.log

# Launch in background; capture PID for R88 daemon
CUDA_VISIBLE_DEVICES=1 nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 1000 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 1e-3 \
    --theta_init 0.01 \
    --kappa_max 0.5 \
    --seed 42 \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 1.0 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --geodesic_kmeans \
    --re_kmeans_every 50 \
    --dead_code_reset_every 0 \
    --dead_code_reset_threshold 0.0 \
    --dead_code_replace_ratio 0.1 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $CKPT_DIR \
    --kappa_log_path $KAPPA_LOG \
    --log_interval 10 \
    --save_every 50 \
    > $ARM_LOG 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > $PID_FILE
echo "[$(date)] A-arm launched with PID $TRAIN_PID, PID file: $PID_FILE" | tee -a $LOG_DIR/task138_launcher.log
echo "[$(date)] Log: $ARM_LOG" | tee -a $LOG_DIR/task138_launcher.log
echo "[$(date)] Launcher exiting (training runs in background, R88 daemon watches PID file)." | tee -a $LOG_DIR/task138_launcher.log