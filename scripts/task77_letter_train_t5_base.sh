#!/bin/bash
# Task #77 — LETTER t5-base 训练 (backbone 升级)
# 沿用 Task #50 RQ-VAE tokenizer (692 tokens) + Trie constrained beam search
# GPU 3 绑定 (单卡, 不用 DDP)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task77_letter_t5base_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #77] LETTER t5-base training launched at $(date) =====" | tee "$LOG_FILE"
echo "Backbone: t5-base (HuggingFace community mirror of google/t5-base, vs Task #50 t5-small)" | tee -a "$LOG_FILE"
echo "RQ-VAE tokenizer: Instruments.index.epoch5000.alpha0.01-beta0.0001.json (692 tokens)" | tee -a "$LOG_FILE"

# 单卡, 不用 DDP (finetune.py 第 4 行已硬编码 CUDA_VISIBLE_DEVICES=3, 此处 export 是冗余保险)
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=0

DATASET=Instruments
OUTPUT_DIR=./ckpt/${DATASET}_t5base/

mkdir -p "$OUTPUT_DIR"

# t5-base 显存比 t5-small 大 ~3.7×, per_device_batch_size 128 → 64
python3 finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --base_model t5-base \
    --per_device_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.epoch5000.alpha0.01-beta0.0001.json \
    --temperature 1.0 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #77] LETTER t5-base training completed at $(date) =====" | tee -a "$LOG_FILE"