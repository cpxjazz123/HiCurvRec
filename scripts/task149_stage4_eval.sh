#!/bin/bash
# Task #149 Stage 4 — Heterogeneous κ HRQ-VAE Test Evaluation
# 2026-07-24
#
# Goal: 验证 Goal #2 下游指标值在合理范围 (paper R@10=0.1315 ±25%)
# 用现有 best ckpt (epoch 49, NDCG@20=0.0977) 跑 Stage 4 test eval
# 即使 Stage 3 后续变, 这个中间版 test 指标足以验证 Goal #2 是否已满足
#
# GPU 2 (R7 空闲 — Task #149 Stage 3 占 GPU 1, Task #151 占 GPU 0)

set -eo pipefail
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task149
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/stage4_test_eval_${TS}.log"
OUTPUT_JSON=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task149_heterokappa_test_eval.json
BEST_CKPT=/home/wlia0047/ar57/wenyu/GeneRec/products/task149/ckpt_heterokappa/Instruments/Jul-24-2026_18-59-00/HG_Rec_best.pth

if [ ! -f "$BEST_CKPT" ]; then
    echo "❌ Stage 3 best ckpt NOT FOUND: $BEST_CKPT"
    exit 1
fi

echo "===== [Task #149 Stage 4] launched at $(date) ====="
echo "GPU: 2 (R7 空闲)"
echo "Best ckpt: $BEST_CKPT"
echo "Code suffix: _t5_hrqvae_heterokappa"
echo "Output JSON: $OUTPUT_JSON"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec

python3 scripts/task149_test_eval_heterokappa.py \
    --ckpt_path "$BEST_CKPT" \
    --code_suffix "_t5_hrqvae_heterokappa" \
    --output_json "$OUTPUT_JSON" \
    --log_file "$LOG_FILE" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #149 Stage 4] done at $(date), exit=$EXIT_CODE ====="