#!/bin/bash
# Task #158 — Issue #57 Gate 0 Stage 3 Arm A: sid_embedding_init='random' (T5 default, baseline)
# 跟 HG-Rec Task #84 baseline 完全一致 — 仅作为 Arm B 对照.
# 200 epoch, lr=5e-4, batch=256, d_model=128, d_ff=1024 (跟 #84 baseline 一致)
# GPU 1 (R7 空闲)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task158_issue57_stage3_random_init_${TS}.log"
PID_FILE="/home/wlia0047/ar57/wenyu/GeneRec/products/task158/_STAGE3_TRAINING_PID"

mkdir -p "$LOG_DIR" /home/wlia0047/ar57/wenyu/GeneRec/products/task158

echo "===== [Task #158 Issue #57 Stage 3 Arm A: random init] launched at $(date) =====" | tee "$LOG_FILE"
echo "sid_embedding_init=random (T5 default, 跟 HG-Rec baseline 完全一致)" | tee -a "$LOG_FILE"
echo "codebook_size=[64,128,256,1] lr=5e-4 batch=256 d_model=128 d_ff=1024 num_layers=6 num_decoder_layers=4" | tee -a "$LOG_FILE"
echo "GPU 1 (R7 空闲)" | tee -a "$LOG_FILE"

# GPU 1 (R7 强制空闲)
export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task158
# R11.4 FIX: scripts/ 子目录 python 不会自动把 HG-Rec/ 加进 sys.path
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task158_issue57_stage3_train.py \
    --batch_size 256 \
    --infer_size 96 \
    --num_epochs 200 \
    --lr 5e-4 \
    --device cuda \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --dropout_rate 0.1 \
    --vocab_size 1025 \
    --pad_token_id 0 \
    --eos_token_id 1024 \
    --feed_forward_proj relu \
    --max_len 20 \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --codebook_size 64 128 256 1 \
    --code_path _A2_t5_hrqvae_poincare.npy \
    --mode train \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task158/ckpt/ \
    --early_stop 20 \
    --disable_early_stop \
    --topk_list 5 10 20 \
    --beam_size 20 \
    --sid_embedding_init random \
    --hyp_c 0.74 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #158 Issue #57 Stage 3 Arm A] training PID: $TRAIN_PID =====" | tee -a "$LOG_FILE"

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #158 Issue #57 Stage 3 Arm A] training completed at $(date), exit code: $EXIT_CODE =====" | tee -a "$LOG_FILE"

rm -f "$PID_FILE"

BEST_CKPT=$(find /home/wlia0047/ar57/wenyu/GeneRec/products/task158/ckpt/Instruments -name "HG_Rec_best.pth" 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    echo "✅ R12 ckpt 落盘: $BEST_CKPT" | tee -a "$LOG_FILE"
else
    echo "❌ R12 ckpt MISSING" | tee -a "$LOG_FILE"
    exit 1
fi