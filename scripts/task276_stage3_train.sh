#!/bin/bash
# Task #276 Stage 3 — A2 curriculum T5-mini training
# 2026-07-29
#
# 输入: products/task276/stage2/A2_t5_hrqvae_poincare.npy (9922, 4) int
#       (Stage 2 已落盘, 1024+ collision 4th-digit dedup)
# 输出: products/task276/stage3/<run_id>/best_collision_model.pth (Stage 3 ckpt, R12)
#       后续 Stage 4 R@10 eval
#
# Recipe: HG-Rec paper-aligned (R11.3):
#   T5-mini, 6 层 encoder + 4 层 decoder, d_model=128, d_ff=1024, 6 heads
#   lr=1e-4, batch=256, infer=96, epochs=200, early_stop=20
#   seed=42 (项目硬约束)
#
# R7 GPU: GPU 1 空闲 (Stage 2 已 done)
# R12: Stage 3 save per epoch + save_total_limit=1 (auto delete old ckpt)
# R8: commit 后立即 push

set -euo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO/HG-Rec

export PYTHONPATH=$REPO/HG-Rec:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/home/wlia0047/ar57_scratch/wenyu/hf_models}

CODE_PATH=_A2_t5_hrqvae_poincare.npy
SAVE_DIR=$REPO/products/task276/stage3
mkdir -p $SAVE_DIR
LOG_FILE=$REPO/logs/task276/stage3_train_$(date +%Y-%m-%d_%H-%M-%S).log
mkdir -p $REPO/logs/task276

GPU=${GPU:-1}
echo "[$(date)] Task #276 Stage 3 T5-mini 训练启动"
echo "[$(date)] code_path=$CODE_PATH"
echo "[$(date)] save_dir=$SAVE_DIR"
echo "[$(date)] GPU=$GPU"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task276_stage3 \
CUDA_VISIBLE_DEVICES=$GPU \
timeout 7200 python3 -u $REPO/scripts/task84_hgrec_stage3_train.py \
    --batch_size 256 --infer_size 96 \
    --num_epochs 200 --lr 1e-4 \
    --num_layers 6 --num_decoder_layers 4 \
    --d_model 128 --d_ff 1024 --num_heads 6 --d_kv 64 \
    --dropout_rate 0.1 \
    --vocab_size 1025 --pad_token_id 0 --eos_token_id 0 \
    --feed_forward_proj relu --max_len 20 \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --codebook_size 64 128 256 1 \
    --code_path $CODE_PATH \
    --mode train \
    --log_path $REPO/logs/task276/ \
    --seed 42 \
    --save_path $SAVE_DIR/ \
    --early_stop 20 \
    --topk_list 5 10 20 \
    --beam_size 20 \
    --device cuda:0 \
    > $LOG_FILE 2>&1

EXIT=$?
echo "[$(date)] exit code=$EXIT"
echo "[$(date)] best_collision_model.pth check:"
find $SAVE_DIR -name "best_collision*.pth" 2>/dev/null | head -3