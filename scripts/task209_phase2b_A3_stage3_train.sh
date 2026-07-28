#!/bin/bash
# Task #209 Phase 2b — A3 Stage 3 T5-mini 训练.
# 输入: task209 A3 SID (Instruments_t5_rqvae_task209_A3.npy, 99.97% 坍缩, 接近 identity 映射)
# 输出: products/task209/t5small_A3/HG_Rec_best.pth (R12 force-save)
# GPU 1 (R7: 4 卡空闲, A0/A3 分配到独立卡)
#
# 跟 task181 stage 3 唯一区别: code_path 改 _t5_rqvae_task209_A3.npy, save_path 改 task209/t5small_A3/.

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task209_phase2b_A3
mkdir -p $TRITON_CACHE_DIR

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task209/t5small_A3
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task209

CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task209_A3.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Phase 2a not complete"
    exit 1
fi
echo "✅ A3 Stage 2 codebook found: $CODE_FILE"

echo "===== [Task #209 Phase 2b A3 Stage 3] T5-mini 训练 at $(date) ====="

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_task209_A3.npy \
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
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task209/t5small_A3/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/

echo "===== A3 Stage 3 训练完成 at $(date) ====="
