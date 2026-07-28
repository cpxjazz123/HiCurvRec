#!/bin/bash
# Task #229 Gate 1 — clean single-variable c=1 Stage 1 (BASELINE REFERENCE)
# 干净配置: kappa_mode=fixed, --curvatures "1.0,1.0,1.0", r_target=None 默认, 无 w_div/w_angular
# 200 epoch, batch=1024, layers=[512,256,128,64], num_emb_list=[64,128,256], e_dim=32
# GPU 0 (R7 空闲)
# R12 ckpt 强制保存 (Trainer 每 eval_step epoch 保存 best_loss/best_collision + epoch)

set -euo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

SAVE_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task229/gate1_c1_baseline
LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task229
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/gate1_c1_stage1_${TS}.log
PID_FILE=$SAVE_DIR/_TRAINING_PID

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "[Task #229 Gate 1 c=1] launching baseline (clean poincare, c=1.0 fixed) on cuda:0 ..." | tee "$LOG_FILE"
echo "Config: kappa_mode=fixed --curvatures '1.0,1.0,1.0' --num_emb_list 64 128 256 --e_dim 32 --epochs 200 --batch_size 1024" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task229_g1_c1
mkdir -p "$TRITON_CACHE_DIR"

python3 -u train_hrqvae.py \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --ckpt_dir "$SAVE_DIR" \
    --loss_type poincare \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --curvatures 1.0,1.0,1.0 \
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
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "[Task #229 Gate 1 c=1] PID=$TRAIN_PID, log=$LOG_FILE" | tee -a "$LOG_FILE"