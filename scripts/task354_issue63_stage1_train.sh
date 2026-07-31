#!/bin/bash
# Task #354 — Issue #63 Stage 1 训练 (vanilla FreeCurvHRQVAE, GPU 0)
# Issue #63 Gate 1 spec: 200 epoch, USAGE-KILL @ ep 30 if min_util < 0.9
# GPU 0 (R7 全部空闲)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/task354_issue63_stage1_train_${TS}.log"
PID_FILE="/home/wlia0047/ar57/wenyu/GeneRec/products/task354/_STAGE1_TRAINING_PID"

mkdir -p "$LOG_DIR" /home/wlia0047/ar57/wenyu/GeneRec/products/task354

echo "===== [Task #354 Issue #63 Stage 1] Vanilla κ-decouple training launched at $(date) =====" | tee "$LOG_FILE"
echo "num_emb_list=[64,128,256] e_dim=32 kappa_max=2.0 layers=[512,256,128] sk_eps=[0,0,0]" | tee -a "$LOG_FILE"
echo "USAGE-KILL @ ep 30 if min_util < 0.9" | tee -a "$LOG_FILE"
echo "GPU 0 (R7 全部空闲)" | tee -a "$LOG_FILE"

# Stage 1 prep
ITEM_EMB=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet
if [ ! -f "$ITEM_EMB" ]; then
    echo "❌ $ITEM_EMB NOT FOUND" | tee -a "$LOG_FILE"
    exit 1
fi

# GPU 0 (R7 强制空闲)
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task354

# Issue #63 Stage 1 训练 (vanilla κ-decouple)
python3 scripts/task354_issue63_stage1_train.py \
    --data_path "$ITEM_EMB" \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task354/ckpt/Instruments \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --M 1 \
    --kappa_max 2.0 \
    --layers 512 256 128 \
    --epochs 200 \
    --batch_size 1024 \
    --lr 1e-3 \
    --loss_type mse \
    --beta 0.25 \
    --kmeans_init True \
    --kmeans_iters 10 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 3 \
    --eval_step 5 \
    --num_workers 4 \
    --device cuda:0 \
    --seed 42 \
    --usage_kill_epoch 30 \
    --usage_kill_threshold 0.9 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #354 Issue #63 Stage 1] training PID: $TRAIN_PID =====" | tee -a "$LOG_FILE"

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #354 Issue #63 Stage 1] training completed at $(date), exit code: $EXIT_CODE =====" | tee -a "$LOG_FILE"

# 清理 PID file
rm -f "$PID_FILE"

# R12 验证 ckpt
if [ -f /home/wlia0047/ar57/wenyu/GeneRec/products/task354/ckpt/Instruments/best_loss_model.pth ]; then
    echo "✅ R12 ckpt 落盘 (best_loss)" | tee -a "$LOG_FILE"
elif [ -f /home/wlia0047/ar57/wenyu/GeneRec/products/task354/ckpt/Instruments/killed_model.pth ]; then
    echo "⚠️ USAGE-KILL ckpt 落盘" | tee -a "$LOG_FILE"
else
    echo "❌ R12 ckpt MISSING" | tee -a "$LOG_FILE"
    exit 1
fi