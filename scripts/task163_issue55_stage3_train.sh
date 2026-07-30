#!/bin/bash
# Task #163 — Issue #55 Stage 3 T5 training (α_l + scale_l SID)
# 跟 Task #158 (#57 Arm A random init) 完全一致, 仅 code_path 改为 issue55 SID
# vocab_size=10380 覆盖 4th-digit dedup counter 最大值
# GPU 3 (R7 强制空闲)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task163_issue55_stage3_train_${TS}.log"
PID_FILE="/home/wlia0047/ar57/wenyu/GeneRec/products/task163/_STAGE3_TRAINING_PID"

mkdir -p "$LOG_DIR" /home/wlia0047/ar57/wenyu/GeneRec/products/task163

echo "===== [Task #163 Issue #55 Stage 3: α_l + scale_l SID] launched at $(date) =====" | tee "$LOG_FILE"
echo "code_path=issue55 SID (mode collapse: 3-digit unique 0.01%, dedup counter up to 9921)" | tee -a "$LOG_FILE"
echo "vocab_size=10380 eos_token_id=10379 codebook_size=[64,128,256,1] lr=5e-4 batch=256 d_model=128 d_ff=1024" | tee -a "$LOG_FILE"
echo "GPU 3 (R7 空闲)" | tee -a "$LOG_FILE"

# GPU 3 (R7 强制空闲)
export CUDA_VISIBLE_DEVICES=3
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task163
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
    --vocab_size 10380 \
    --pad_token_id 0 \
    --eos_token_id 10379 \
    --feed_forward_proj relu \
    --max_len 20 \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --codebook_size 64 128 256 1 \
    --code_path _issue55_t5_hrqvae_poincare.npy \
    --mode train \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task163/ckpt/ \
    --early_stop 20 \
    --disable_early_stop \
    --topk_list 5 10 20 \
    --beam_size 20 \
    --sid_embedding_init random \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #163 Issue #55 Stage 3] training PID: $TRAIN_PID =====" | tee -a "$LOG_FILE"

wait $TRAIN_PID
EXIT_CODE=$?
echo "===== [Task #163 Issue #55 Stage 3] training completed at $(date), exit code: $EXIT_CODE =====" | tee -a "$LOG_FILE"

rm -f "$PID_FILE"

BEST_CKPT=$(find /home/wlia0047/ar57/wenyu/GeneRec/products/task163/ckpt/Instruments -name "HG_Rec_best.pth" 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    echo "✅ R12 ckpt 落盘: $BEST_CKPT" | tee -a "$LOG_FILE"
else
    echo "❌ R12 ckpt MISSING" | tee -a "$LOG_FILE"
    exit 1
fi
