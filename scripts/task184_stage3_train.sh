#!/bin/bash
# Task #184 Stage 3 — T5-small training (uses product_manifold codebook)
# 启动 GPU 1 (Task #185 训练在 GPU 0, GPU 1 空闲)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task184_stage3

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task184
mkdir -p /fs04/ar57/wenyu/GeneRec/logs/task184

# 验证 Stage 2 codebook
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_product_manifold.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND"
    exit 1
fi
echo "✅ Stage 2 codebook: $CODE_FILE"

echo "===== [Task #184 Stage 3] T5-small training at $(date) ====="

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --batch_size 1024 \
    --infer_size 96 \
    --lr 1e-4 \
    --num_epochs 200 \
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
    --device cuda:1 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task184/t5small_product_manifold/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --code_path _t5_rqvae_product_manifold.npy \
    2>&1 | tee /fs04/ar57/wenyu/GeneRec/logs/task184/stage3_train.out

echo "===== [Task #184 Stage 3] completed at $(date) ====="
