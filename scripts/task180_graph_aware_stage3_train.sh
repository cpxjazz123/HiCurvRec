#!/bin/bash
# Task #180 Stage 3 — T5-small 5.5M on graph-aware SID tensor.
#
# 与 Task #178 Stage 3 唯一变量: --code_path 用 graph-aware SID tensor
#   (Instruments_t5_rqvae_graph_aware.npy, 由 LightGCN + 修正版 HRQ-VAE 生成)
# 其他完全跟 Task #178 一致, 保证单变量对照.

set -euo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

# R7 GPU 分配: GPU 2 完全空闲 (util 0%, mem 3 MiB)
# GPU 0 被 #178 Stage 3 占 (PID 424514, util 94%)
# GPU 1 被 #179 Stage 3 占 (PID 448827, util 82%)
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task180_stage3
mkdir -p "$TRITON_CACHE_DIR"

CODE_PATH="_t5_rqvae_graph_aware.npy"
DATASET_NAME="Instruments"
DATASET_PATH="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/"
SAVE_PATH="/home/wlia0047/ar57/wenyu/GeneRec/products/task180/t5small_graph_aware/Jul-25-2026_05-01-XX"
LOG_PATH="/home/wlia0047/ar57/wenyu/GeneRec/logs/task180"
mkdir -p "$SAVE_PATH"
mkdir -p "$LOG_PATH"

python3 scripts/task84_hgrec_stage3_train.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_path "$DATASET_PATH" \
    --code_path "$CODE_PATH" \
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
    --save_path "$SAVE_PATH" \
    --log_path "$LOG_PATH" \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96