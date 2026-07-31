#!/bin/bash
# Task #402 / Stage 3 T5-mini 训练 (基于 task401 ckpt 派生的 _t5_rqvae_task402.npy)
# Date: 2026-07-31
# GPU: 1 (R7 空闲, parallel to task396b GPU 0)
# 200 epoch 跟 task396 baseline 同, 验证 task401 训练时长杠杆下游是否成立
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task402_stage3
mkdir -p $TRITON_CACHE_DIR
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream/ckpt

cd /home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream
nohup python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_task402.npy \
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
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task402_ckpt_downstream/ckpt/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 > task402_stage3_nohup.out 2>&1 &
TRAIN_PID=$!
disown
echo $TRAIN_PID > _STAGE3_PID
echo "Task #402 Stage 3 PID: $TRAIN_PID"