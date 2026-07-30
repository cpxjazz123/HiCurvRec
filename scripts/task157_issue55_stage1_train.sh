#!/bin/bash
# Task #157 — Issue #55 Stage 1 训练 (Riemannian AdamW + per-layer s_l, GPU 3)
# R11.3 自决: 1000 epoch, num_emb_list=[64,128,256], kappa_fixed=1.0, RiemannianAdamW
# GPU 3 (R7 空闲)
# R12 ckpt 强制保存

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task157_issue55_stage1_train_${TS}.log"
PID_FILE="/home/wlia0047/ar57/wenyu/GeneRec/products/task157/_STAGE1_TRAINING_PID"

mkdir -p "$LOG_DIR" /home/wlia0047/ar57/wenyu/GeneRec/products/task157

echo "===== [Task #157 Issue #55 Stage 1] Riemannian AdamW training launched at $(date) =====" | tee "$LOG_FILE"
echo "kappa_fixed=1.0 alpha_init=0.5 num_emb_list=[64,128,256] e_dim=32 sk_epsilons=[0,0,0.000]" | tee -a "$LOG_FILE"
echo "Optimizer: RiemannianAdamW (codebook) + AdamW (others)" | tee -a "$LOG_FILE"
echo "GPU 3 (R7 空闲)" | tee -a "$LOG_FILE"

ITEM_EMB=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet
if [ ! -f "$ITEM_EMB" ]; then
    echo "❌ $ITEM_EMB NOT FOUND — Stage 1 prep not complete" | tee -a "$LOG_FILE"
    exit 1
fi

# GPU 3 (R7 强制空闲)
export CUDA_VISIBLE_DEVICES=3
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task157

python3 scripts/task157_issue55_stage1_train.py \
    --data_path "$ITEM_EMB" \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task157/ckpt/Instruments \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --M 1 \
    --kappa_fixed 1.0 \
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
echo "===== [Task #157 Issue #55 Stage 1] training PID: $TRAIN_PID =====" | tee -a "$LOG_FILE"

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #157 Issue #55 Stage 1] training completed at $(date), exit code: $EXIT_CODE =====" | tee -a "$LOG_FILE"

rm -f "$PID_FILE"

BEST_LOSS=$(ls /home/wlia0047/ar57/wenyu/GeneRec/products/task157/ckpt/Instruments/*/best_loss_model.pth 2>/dev/null | head -1)
if [ -n "$BEST_LOSS" ]; then
    echo "✅ R12 ckpt 落盘: $BEST_LOSS" | tee -a "$LOG_FILE"
else
    echo "❌ R12 ckpt MISSING" | tee -a "$LOG_FILE"
    exit 1
fi