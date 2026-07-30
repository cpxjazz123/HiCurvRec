#!/bin/bash
# Task #327 / K=256 + Issue #30 — Stage 3 T5-mini 200 epoch training
# 输入: SID at HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k256_issue30.npy
# 输出: products/task327/t5mini_k256_issue30/best_ckpt.pth

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

LOG_DIR=$REPO/logs/task327
mkdir -p "$LOG_DIR"

SAVE_PATH=$REPO/products/task327/t5mini_k256_issue30
mkdir -p $SAVE_PATH

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage3_t5mini_${TS}.log

echo "[$(date)] Launching Task #327 Stage 3 T5-mini"
echo "  --code_path _t5_rqvae_k256_issue30.npy"
echo "  --codebook_size 256 128 256 1"
echo "  --num_epochs 200, beam_size 50 (Issue #30 K=50 amplifier)"

CUDA_VISIBLE_DEVICES=1 nohup "$PYTHON_BIN" -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_rqvae_k256_issue30.npy \
    --codebook_size 256 128 256 1 \
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
    --save_path $SAVE_PATH \
    --log_path $LOG_DIR/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 50 \
    --infer_size 96 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > $REPO/products/task327/_TRAINING_PID_STAGE3
echo "[$(date)] Stage 3 PID=$TRAIN_PID, GPU=1"
echo ""
echo "===== Task #327 / Stage 3 T5-mini Launched ====="
echo "PID=$TRAIN_PID (GPU 1, K=256 + Issue #30, 200 epoch, beam=50)"
echo "Log: $LOG_FILE"
echo "Expected runtime: ~50-80 min"