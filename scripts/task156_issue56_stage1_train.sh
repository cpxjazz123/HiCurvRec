#!/bin/bash
# Task #156 — Issue #56 Stage 1 训练 (α_l mixed_curv_dist, GPU 2)
# R11.3 自决: 1000 epoch, num_emb_list=[64,128,256], kappa_fixed=0.74, alpha_init=0.5
# GPU 2 (R7 空闲)
# R12 ckpt 强制保存 (每个 eval_step epoch 保存 best_loss + final)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task156_issue56_stage1_train_${TS}.log"
PID_FILE="/home/wlia0047/ar57/wenyu/GeneRec/products/task156/_STAGE1_TRAINING_PID"

mkdir -p "$LOG_DIR" /home/wlia0047/ar57/wenyu/GeneRec/products/task156

echo "===== [Task #156 Issue #56 Stage 1] Mixed-curv α_l training launched at $(date) =====" | tee "$LOG_FILE"
echo "kappa_fixed=0.74 alpha_init=0.5 num_emb_list=[64,128,256] e_dim=32 sk_epsilons=[0,0,0.000]" | tee -a "$LOG_FILE"
echo "GPU 2 (R7 空闲)" | tee -a "$LOG_FILE"

# Stage 1 prep (item_emb.parquet) 已完成 (script task84_hgrec_stage1_prep.py)
ITEM_EMB=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet
if [ ! -f "$ITEM_EMB" ]; then
    echo "❌ $ITEM_EMB NOT FOUND — Stage 1 prep not complete" | tee -a "$LOG_FILE"
    exit 1
fi

# GPU 2 (R7 强制空闲)
export CUDA_VISIBLE_DEVICES=2
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task156

# Issue #56 Stage 1 训练
python3 scripts/task156_issue56_stage1_train.py \
    --data_path "$ITEM_EMB" \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt/Instruments \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --M 1 \
    --kappa_fixed 0.74 \
    --layers 512 256 128 64 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --loss_type mse \
    --beta 0.25 \
    --kmeans_init True \
    --kmeans_iters 10 \
    --sk_epsilons 0.0 0.0 0.000 \
    --sk_iters 3 \
    --eval_step 5 \
    --num_workers 4 \
    --device cuda:0 \
    --seed 42 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #156 Issue #56 Stage 1] training PID: $TRAIN_PID =====" | tee -a "$LOG_FILE"

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #156 Issue #56 Stage 1] training completed at $(date), exit code: $EXIT_CODE =====" | tee -a "$LOG_FILE"

# 清理 PID file
rm -f "$PID_FILE"

# R12 验证 ckpt 是否落盘
BEST_LOSS=$(ls /home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt/Instruments/*/best_loss_model.pth 2>/dev/null | head -1)
if [ -n "$BEST_LOSS" ]; then
    echo "✅ R12 ckpt 落盘: $BEST_LOSS" | tee -a "$LOG_FILE"
else
    echo "❌ R12 ckpt MISSING" | tee -a "$LOG_FILE"
    exit 1
fi