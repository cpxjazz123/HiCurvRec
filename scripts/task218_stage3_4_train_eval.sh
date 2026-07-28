#!/usr/bin/env bash
# Task #218 Stage 3 + 4 — T5-mini 训练 + 评估 per_codeword κ SID codebook
# 复用 task84 Stage 3 脚本 + task84 Stage 4 eval 模式 (R12 + R11 验证)
# GPU 0 (跟 Stage 1 一致)

set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

# Verify Stage 2 codebook exists
CODE_FILE=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_pck_spread.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Stage 2 not complete"
    exit 1
fi

LOG_DIR=$REPO/logs/task218
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage3_4_${TS}.log
mkdir -p $LOG_DIR $REPO/products/task218/ckpt_hgrec_pck_spread

echo "===== [Task #218 Stage 3+4] per_codeword κ SID launched at $(date) =====" | tee $LOG_FILE
echo "code_path=$CODE_FILE" | tee -a $LOG_FILE

# Stage 3: T5-mini 训练 (跟 task84 一致, 但用 task218 code_path)
python3 $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_pck_spread.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path $REPO/products/task218/ckpt_hgrec_pck_spread/ \
    --log_path $LOG_DIR/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task218/_STAGE3_TRAINING_PID
echo "[Stage 3] PID=$TRAIN_PID" | tee -a $LOG_FILE
wait $TRAIN_PID

EXIT_CODE=$?
echo "[Stage 3] exit code=$EXIT_CODE" | tee -a $LOG_FILE

# R12 ckpt 验证
BEST_CKPT=$(ls $REPO/products/task218/ckpt_hgrec_pck_spread/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "❌ R12 best ckpt MISSING"
    exit 1
fi
echo "✅ Stage 3 best ckpt: $BEST_CKPT" | tee -a $LOG_FILE

# Stage 4: 评估 R@10
echo "[Stage 4] evaluation ..." | tee -a $LOG_FILE
python3 $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_pck_spread.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 1 \
    --batch_size 256 \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode evaluation \
    --save_path $REPO/products/task218/ckpt_hgrec_pck_spread/ \
    --log_path $LOG_DIR/ \
    --seed 42 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee -a $LOG_FILE

rm -f $REPO/products/task218/_STAGE3_TRAINING_PID
echo "===== [Task #218 Stage 3+4] all complete at $(date) =====" | tee -a $LOG_FILE
