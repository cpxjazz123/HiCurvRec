#!/bin/bash
# Task #150 Stage 4 — LETTER paper-aligned test inference
# 2026-07-24
#
# 训练已完 (epoch 4.0 / step 16480 / train_loss 26.76 / eval_loss 11.56).
# 本阶段: 用最终 model.safetensors 跑 Instruments test set, 输出 hit@5/10/20 + ndcg@5/10.
#
# R7 GPU: GPU 2 (Task #149 Stage 3 占 GPU 1, Task #151 P5-CID 占 GPU 0)
# hit@10 = Recall@10 (per evaluate.py hit_k definition)

set -eo pipefail
export CUDA_VISIBLE_DEVICES=2
export USE_TF=0                  # FIX Task #139 Keras 3 crash
export WANDB_DISABLED=true       # FIX wandb ImportError
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task150_stage4

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task150
mkdir -p "$LOG_DIR"

CKPT_PATH=/home/wlia0047/ar57/wenyu/GeneRec/products/task150/ckpt_paper_aligned
RESULTS_FILE=/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task150_letter_test_metrics.json

if [ ! -f "$CKPT_PATH/model.safetensors" ]; then
    echo "❌ model.safetensors NOT FOUND: $CKPT_PATH/model.safetensors"
    exit 1
fi

echo "===== [Task #150 Stage 4] launched at $(date) ====="
echo "GPU: 2 (R7 空闲)"
echo "Ckpt: $CKPT_PATH/model.safetensors"
echo "Metrics output: $RESULTS_FILE"

cd /home/wlia0047/ar57/wenyu/GeneRec/LETTER/LETTER-TIGER

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/stage4_inference_${TS}.log"

python3 test.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset Instruments \
    --tasks seqrec \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/LETTER/data \
    --index_file .index.json \
    --output_dir "$CKPT_PATH" \
    --results_file "$RESULTS_FILE" \
    --gpu_id 0 \
    --num_beams 20 \
    --test_batch_size 8 \
    --metrics "hit@1,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20" \
    --test_prompt_ids 0 \
    --seed 42 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "===== [Task #150 Stage 4] done at $(date), exit=$EXIT_CODE ====="