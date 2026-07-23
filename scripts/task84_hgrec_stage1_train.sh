#!/bin/bash
# Task #84 Stage 1 — HRQ-VAE 训练 (loss_type='poincare', GPU 1)
# R11.3 自决: 1000 epoch, num_emb_list=[64,128,256], e_dim=32, sk_epsilons=[0,0,0.000]
# GPU 1 (R7 空闲 — S3Rec GPU 0, P5-SID GPU 2)
# R12 ckpt 强制保存 (Trainer 每 eval_step epoch 保存 best_loss/best_collision + epoch, save_limit=5)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task84_hgrec_stage1_train_${TS}.log"
PID_FILE="/home/wlia0047/ar57/wenyu/GeneRec/products/task84/_STAGE1_TRAINING_PID"

mkdir -p "$LOG_DIR" /home/wlia0047/ar57/wenyu/GeneRec/products/task84

echo "===== [Task #84 Stage 1] HRQ-VAE training launched at $(date) =====" | tee "$LOG_FILE"
echo "loss_type=poincare num_emb_list=[64,128,256] e_dim=32 sk_epsilons=[0,0,0.000] layers=[512,256,128,64]" | tee -a "$LOG_FILE"
echo "GPU 1 (S3Rec=GPU 0, P5-SID=GPU 2)" | tee -a "$LOG_FILE"

# Stage 1 prep (item_emb.parquet) 已完成 (script task84_hgrec_stage1_prep.py log)
ITEM_EMB=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet
if [ ! -f "$ITEM_EMB" ]; then
    echo "❌ $ITEM_EMB NOT FOUND — Stage 1 prep not complete" | tee -a "$LOG_FILE"
    exit 1
fi

# GPU 1 (R7 强制空闲)
export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task84

# R11.3: 启动 HRQ-VAE 训练 (loss_type=poincare 是 HG-Rec 默认, 但显式指定)
python3 train_hrqvae.py \
    --data_path "$ITEM_EMB" \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments \
    --loss_type poincare \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --sk_epsilons 0.0 0.0 0.000 \
    --layers 512 256 128 64 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --eval_step 5 \
    --num_workers 4 \
    --device cuda:0 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #84 Stage 1] HRQ-VAE training PID: $TRAIN_PID =====" | tee -a "$LOG_FILE"

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #84 Stage 1] HRQ-VAE training completed at $(date), exit code: $EXIT_CODE =====" | tee -a "$LOG_FILE"

# 清理 PID file
rm -f "$PID_FILE"

# R12 验证 ckpt 是否落盘
BEST_LOSS=$(ls /home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/*/best_loss_model.pth 2>/dev/null | head -1)
if [ -n "$BEST_LOSS" ]; then
    echo "✅ R12 ckpt 落盘: $BEST_LOSS" | tee -a "$LOG_FILE"
else
    echo "❌ R12 ckpt MISSING" | tee -a "$LOG_FILE"
    exit 1
fi
