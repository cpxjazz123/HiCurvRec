#!/bin/bash
# Task #276 Stage 4 — A2 curriculum R@10 eval (test set)
# 2026-07-29
#
# 输入: products/task276/stage3/Instruments/Jul-29-2026_10-50-15/HG_Rec_best.pth
#       products/task276/stage2/A2_t5_hrqvae_poincare.npy
# 输出: products/task276/stage4/eval_metrics.json
#
# GO/NO-GO: R@10 > HG-Rec baseline 0.1020
# R7: GPU 0 idle (Stage 3 占 GPU 1)
# R12: best ckpt 来自 Stage 3 R12 save

set -euo pipefail
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO

CKPT=$REPO/products/task276/stage3/Instruments/Jul-29-2026_10-50-15/HG_Rec_best.pth
CODE_PATH=$REPO/products/task276/stage2/A2_t5_hrqvae_poincare.npy
OUTPUT=$REPO/products/task276/stage4/eval_metrics.json
LOG=$REPO/logs/task276/stage4_eval_$(date +%Y-%m-%d_%H-%M-%S).log
mkdir -p $REPO/logs/task276

GPU=${GPU:-0}
echo "[$(date)] Task #276 Stage 4 eval 启动 GPU=$GPU"
echo "[$(date)] ckpt=$CKPT"
echo "[$(date)] code_path=$CODE_PATH"

TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task276_stage4 \
CUDA_VISIBLE_DEVICES=$GPU \
timeout 1800 python3 -u $REPO/scripts/task276_stage4_eval.py \
    --ckpt_path "$CKPT" \
    --code_path "$CODE_PATH" \
    --dataset_name Instruments \
    --dataset_path $REPO/HG-Rec/dataset/ \
    --device cuda:0 \
    --batch_size 96 \
    --beam_size 20 \
    --max_len 20 \
    --seed 42 \
    --topk_list 5 10 20 \
    --output_path "$OUTPUT" \
    > $LOG 2>&1

EXIT=$?
echo "[$(date)] exit code=$EXIT"
echo "[$(date)] metrics:"
cat "$OUTPUT" 2>/dev/null
