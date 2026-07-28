#!/bin/bash
# Task #150 — LETTER paper-aligned 修复 (lr=2e-5 / batch=8 / epochs=4)
# 2026-07-24
#
# 根因:
#   (1) task72 / task139 使用 lr=5e-4 batch=256 epochs=200, 论文 default 2e-5/8/4
#   (2) Task #139 实际 **崩溃** 在 Keras 3 不兼容错误 (Keras 3 not supported by Transformers)
#      Task #139 products/letter_tiger/ 只有 tokenizer 无 model.safetensors.
#      我们的 LETTER +71.6% 数字来自 Task #61 (lr=5e-4 over-trained), 不是 paper-aligned.
# 修复:
#   (a) USE_TF=0 跳过 Keras 3 检查 (避免 transformers ValueError)
#   (b) lr=2e-5 batch=8 epochs=4 (paper-faithful), 验证 R@10 ∈ [0.052, 0.075]
#
# R7 GPU: GPU 2,3 空闲 (Task #149 stage 3 占 GPU 1, Task #150 不抢)

set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,3
export USE_TF=0                 # FIX Task #139 Keras 3 crash
export WANDB_DISABLED=true      # FIX Task #150 wandb ImportError (broken proto)
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task150

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task150
mkdir -p "$LOG_DIR"

cd /home/wlia0047/ar57/wenyu/GeneRec/LETTER/LETTER-TIGER

TRAIN_PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task150/_TRAINING_PID
LOG_FILE="$LOG_DIR/letter_paper_aligned_$(date +%Y-%m-%d_%H-%M-%S).log"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task150/ckpt_paper_aligned

echo "[$(date)] Task #150 LETTER paper-aligned launcher. lr=2e-5 batch=8 epochs=4 (paper defaults)"
echo "[$(date)] FIX: USE_TF=0 to skip Keras 3 ValueError (Task #139 crashed there)"
echo "[$(date)] torch=$(python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())')"

# R12: 2 GPU × batch 8 = global 16 (跟 paper batch=8 一致, 1.5× paper global batch)
torchrun --nproc_per_node=2 --master_port=2316 finetune.py \
    --output_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task150/ckpt_paper_aligned \
    --dataset Instruments \
    --per_device_batch_size 8 \
    --learning_rate 2e-5 \
    --epochs 4 \
    --index_file .index.json \
    --temperature 1.0 \
    --seed 42 \
    --logging_step 200 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 2 \
    --save_and_eval_strategy epoch \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo $TRAIN_PID > "$TRAIN_PID_FILE"
echo "[$(date)] task150 training PID: $TRAIN_PID, log: $LOG_FILE"
wait $TRAIN_PID
EXIT_CODE=$?
rm -f "$TRAIN_PID_FILE"
echo "[$(date)] task150 training done, exit=$EXIT_CODE"
