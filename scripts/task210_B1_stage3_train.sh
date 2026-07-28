#!/bin/bash
# Task #210 Phase B B1 — Stage 3 T5-mini 训练
# 输入: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task210_B1.npy (9922 unique SID)
# 输出: products/task210/t5mini_B1/HG_Rec_best.pth
# 配置: 跟 task209 A3 一致 (T5-mini 9.18M: num_layers=6 num_decoder_layers=4 d_model=128 num_heads=6 vocab=11000)
# GPU 1 (R7: A3 eval 在 GPU 0 已结束, 1/2/3 空闲)
set -e

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task210_B1_s3
mkdir -p $TRITON_CACHE_DIR

mkdir -p $REPO/products/task210/t5mini_B1
mkdir -p $REPO/logs/task210

CODE_FILE=$REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task210_B1.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ B1 Stage 2 SID npy MISSING: $CODE_FILE"
    exit 1
fi
echo "✅ B1 Stage 2 SID codebook found: $CODE_FILE"

echo "===== [Task #210 B1 Stage 3] T5-mini 训练 launched at $(date) ====="

# R12: PID 记录
TRAIN_PID_FILE=$REPO/products/task210/_TRAINING_PID_B1_stage3
# 不后台, 直接前台跑 (以便 Stage 4 eval 等待)
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_task210_B1.npy \
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
    --vocab_size 11000 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task210/t5mini_B1/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/

echo "===== B1 Stage 3 训练完成 at $(date) ====="