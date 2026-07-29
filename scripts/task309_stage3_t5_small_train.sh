#!/bin/bash
# Task #309 — Stage 3 T5-mini → T5-small upgrade (60M params)
# 目的: 验证 K14 (Stage 3/4 训练协议是真 R@10 杠杆) 在容量扩展维度
# 沿用 #30 Stage 2 SID (per-layer Codebook Transforms)
# 2026-07-30
#
# R7: GPU 1 (R7 空闲, 4×L40S)
# R12: save best_ckpt at every epoch (save_limit=1 fork)
# R14: 跟 K14 联立

set -o pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task309

LOG_DIR=$REPO/logs/task309
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage3_train_${TS}.log
PID_FILE=$REPO/products/task309/_STAGE3_TRAINING_PID

mkdir -p $REPO/products/task309/ckpt_hgrec_t5_small/Instruments

echo "===== [Task #309 Stage 3 T5-small] launched at $(date) =====" | tee $LOG_FILE
echo "code_path = _t5_hrqvae_issue30_per_layer_transforms.npy (#30 GO config)" | tee -a $LOG_FILE
echo "T5 config: d_model=512 d_ff=2048 num_heads=8 num_layers=6 num_decoder_layers=6 (T5-small standard)" | tee -a $LOG_FILE
echo "GPU 1 (R7 空闲)" | tee -a $LOG_FILE
echo "Expected params: ~60M (T5-small)" | tee -a $LOG_FILE

CODE_FILE=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue30_per_layer_transforms.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND" | tee -a $LOG_FILE
    exit 1
fi

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --code_path _t5_hrqvae_issue30_per_layer_transforms.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 6 \
    --num_decoder_layers 6 \
    --d_model 512 \
    --d_ff 2048 \
    --num_heads 8 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path $REPO/products/task309/ckpt_hgrec_t5_small/ \
    --log_path $REPO/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 50 \
    --infer_size 96 \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "===== [Task #309 Stage 3 T5-small] Training PID: $TRAIN_PID =====" | tee -a $LOG_FILE
echo "Log: $LOG_FILE" | tee -a $LOG_FILE
echo "Expected runtime: ~4-5h" | tee -a $LOG_FILE
