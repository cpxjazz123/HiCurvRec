#!/bin/bash
# Task #312 / Issue #35 — Stage 3 T5-mini 200 epoch 训练 on r_l identity + s_l=[2,2,2] SID
# R11.5 决策: 沿用 task301 #30 Stage 3 配置 (T5-mini, 200 epoch, early_stop=20, beam=50 K14)
# GPU 0 跟 task309 GPU 1 不冲突
# 2026-07-30

set -o pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task312_gate3

LOG_DIR=$REPO/logs/task312
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log
PID_FILE=$REPO/products/task312/_STAGE3_TRAINING_PID

mkdir -p $REPO/products/task312/ckpt_hgrec_issue35/Instruments

echo "===== [Task #312 / Issue #35 Gate 3 Stage 3] T5-mini 200 epoch launched at $(date) =====" | tee $LOG_FILE
echo "code_path = _t5_hrqvae_issue35_rl_identity_sl22.npy (Stage 2 output)" | tee -a $LOG_FILE
echo "GPU 0 (R7 空闲, task309 Stage 3 用 GPU 1)" | tee -a $LOG_FILE

CODE_FILE=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue35_rl_identity_sl22.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_issue35_rl_identity_sl22.npy \
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
    --save_path $REPO/products/task312/ckpt_hgrec_issue35/ \
    --log_path $REPO/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 50 \
    --infer_size 96 \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #312 / Issue #35 Gate 3 Stage 3] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE
