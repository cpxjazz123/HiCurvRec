#!/bin/bash
# Task #239 Gate 2 A2 — c=[30,3,3] REVERSED proposal (L0 显著提曲率)
# GPU 1 (R7: 全空闲)
set -euo pipefail
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

SAVE_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task239/gate2_c30_3_3
LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task239
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/gate2_c30_3_3_${TS}.log
PID_FILE=$SAVE_DIR/_TRAINING_PID

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "[Task #239 Gate 2 A2 c=[30,3,3]] launching on cuda:1 ..." | tee "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task239_g2_a2
mkdir -p "$TRITON_CACHE_DIR"

python3 -u train_hrqvae.py \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_dir "$SAVE_DIR" \
    --loss_type poincare \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --curvatures 30.0,3.0,3.0 \
    --sk_epsilons 0.0 0.0 0.000 \
    --layers 512 256 128 64 \
    --epochs 200 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --eval_step 5 \
    --num_workers 4 \
    --save_limit 5 \
    --device cuda:0 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "[Task #239 Gate 2 A2] PID=$TRAIN_PID, log=$LOG_FILE" | tee -a "$LOG_FILE"
