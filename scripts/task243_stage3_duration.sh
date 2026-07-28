#!/bin/bash
# Task #243 — Stage 3 训练时长单变量 (200 ep vs 400 ep)
# 目的: 测试 Stage 3 num_epochs 是否是 R@10 真正变量 (per Task #237 §后续建议).
# 协议: 同一 SID (task84 baseline), 不同 epoch 上限, 其它完全相同.
# Wall: ~88 min 并行 (400 ep bottleneck) on GPU 1 + GPU 2.
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

# === Setup shared ===
mkdir -p $REPO/products/task243 $REPO/logs/task243
SID_FILE="_t5_rqvae_code_default.npy"  # task84 baseline Stage 2 SID
DATA_DIR=$REPO/HG-Rec/dataset

# === Run 1: epoch=200 on GPU 1 (baseline repeat, sanity check) ===
SAVE_DIR_200=$REPO/products/task243/t5mini_epoch200
LOG_200=$REPO/logs/task243/stage3_epoch200.out
PID_200_FILE=$REPO/products/task243/_TRAINING_PID_E200
mkdir -p $SAVE_DIR_200
rm -f $LOG_200 $PID_200_FILE

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #243 Stage 3 epoch=200 on GPU 1 ===" | tee $LOG_200

CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task243_e200 \
python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments --dataset_path $DATA_DIR/ \
    --code_path $SID_FILE \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 --batch_size 256 --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 --d_model 128 --d_ff 1024 \
    --num_heads 6 --d_kv 64 --vocab_size 1025 --max_len 20 \
    --pad_token_id 0 --eos_token_id 0 \
    --device cuda:0 --mode train \
    --save_path $SAVE_DIR_200 \
    --log_path $REPO/logs/task243/ \
    --seed 42 --early_stop 20 --beam_size 20 --infer_size 96 \
    > $LOG_200 2>&1 &
PID_200=$!
echo $PID_200 > $PID_200_FILE

# === Run 2: epoch=400 on GPU 2 ===
SAVE_DIR_400=$REPO/products/task243/t5mini_epoch400
LOG_400=$REPO/logs/task243/stage3_epoch400.out
PID_400_FILE=$REPO/products/task243/_TRAINING_PID_E400
mkdir -p $SAVE_DIR_400
rm -f $LOG_400 $PID_400_FILE

echo "[$(date +%Y-%m-%d_%H:%M:%S)] === Task #243 Stage 3 epoch=400 on GPU 2 ===" | tee $LOG_400

CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task243_e400 \
python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments --dataset_path $DATA_DIR/ \
    --code_path $SID_FILE \
    --codebook_size 64 128 256 1 \
    --num_epochs 400 --batch_size 256 --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 --d_model 128 --d_ff 1024 \
    --num_heads 6 --d_kv 64 --vocab_size 1025 --max_len 20 \
    --pad_token_id 0 --eos_token_id 0 \
    --device cuda:0 --mode train \
    --save_path $SAVE_DIR_400 \
    --log_path $REPO/logs/task243/ \
    --seed 42 --early_stop 20 --beam_size 20 --infer_size 96 \
    > $LOG_400 2>&1 &
PID_400=$!
echo $PID_400 > $PID_400_FILE

echo "[$(date +%Y-%m-%d_%H:%M:%S)] Both launched: 200 ep PID=$PID_200, 400 ep PID=$PID_400"
