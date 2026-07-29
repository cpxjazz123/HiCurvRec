#!/bin/bash
# Task #300 Issue #29 Gate 3 — Stage 3 T5-mini 200 epoch 训练 (per-layer K_l SID)
# 2026-07-30
#
# 背景: Issue #29 Gate 1+2 PASS. Stage 3 = T5-mini 200 epoch 训练, code_path = issue29_per_layer_k.
#       Stage 4 = load best ckpt 评估 test R@10.
# R11.3: 沿用 task84_hgrec_stage3_train.py + task84_hgrec_stage4_eval 模式
# R7: GPU 0 / GPU 1 (R7 空闲, 4×L40S)
# R12: save_limit=1 (Stage 3 ckpt fork)
# R14: Issue #29 hard-stop at Gate 3 fail

set -o pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO/HG-Rec

# PYTHONPATH for `from model.HG_Rec`
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task300_gate3

LOG_DIR=$REPO/logs/task300
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log
PID_FILE=$REPO/products/task300/_STAGE3_TRAINING_PID

mkdir -p $REPO/products/task300/ckpt_hgrec_issue29/Instruments

echo "===== [Task #300 Issue #29 Gate 3 Stage 3] per-layer K_l SID launched at $(date) =====" | tee $LOG_FILE
echo "code_path = _t5_hrqvae_issue29_per_layer_k.npy (Stage 2 output)" | tee -a $LOG_FILE
echo "codebook_size = 128 64 32 1 (per-layer K_l)" | tee -a $LOG_FILE
echo "GPU 0 (R7 空闲)" | tee -a $LOG_FILE

# Verify Stage 2 codebook exists
CODE_FILE=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue29_per_layer_k.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_issue29_per_layer_k.npy \
    --codebook_size 128 64 32 1 \
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
    --save_path $REPO/products/task300/ckpt_hgrec_issue29/ \
    --log_path $REPO/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #300 Issue #29 Gate 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE
