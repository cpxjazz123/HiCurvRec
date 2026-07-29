#!/bin/bash
# Task #290/#291/#292 — Stage 2/3/4 pipeline wrapper
# 2026-07-29: 3 个 Stage 1 已完成 (FSQ/EMA/Restoration). 立即启动 Stage 2 Sinkhorn → Stage 3 T5-mini → Stage 4 R@10.
# 用法: bash task29x_stage234_pipeline.sh <task_id> <gpu_id>
#   task_id: 290/291/292
#   gpu_id: 0/1/2 (R7 不抢已占卡)

set -e

TASK_ID=${1:-290}
GPU_ID=${2:-0}
REPO=/home/wlia0047/ar57/wenyu/GeneRec
HGREC=$REPO/HG-Rec

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# 确定 quantizer type from task_id
case $TASK_ID in
  290) QUANTIZER=fsq ;;
  291) QUANTIZER=ema ;;
  292) QUANTIZER=restoration ;;
  *) echo "❌ Unknown task_id $TASK_ID"; exit 1 ;;
esac

LOG_DIR=$REPO/logs/task${TASK_ID}
mkdir -p $LOG_DIR $REPO/products/task${TASK_ID}
TS=$(date +%d-%H-%M-%S)
LOG_FILE=$LOG_DIR/stage234_${TS}.log

echo "===== [Task #${TASK_ID} Stage 234] QUANTIZER=${QUANTIZER} GPU=${GPU_ID} at $(date) =====" | tee "$LOG_FILE"

# GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task${TASK_ID}
mkdir -p $TRITON_CACHE_DIR

# 找 Stage 1 best ckpt
BEST_CKPT=$(ls -t $REPO/products/task${TASK_ID}/hrqvae_*/Instruments/*/best_loss_model.pth 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "❌ Stage 1 best_loss_model.pth NOT FOUND for task${TASK_ID}" | tee -a "$LOG_FILE"
    exit 1
fi
echo "Stage 1 best ckpt: $BEST_CKPT" | tee -a "$LOG_FILE"

# ========== Stage 2 Sinkhorn Inference ==========
STAGE2_OUTPUT=$HGREC/dataset/Instruments/Instruments_t5_hrqvae_${QUANTIZER}.npy
echo "" | tee -a "$LOG_FILE"
echo "===== [Task #${TASK_ID} Stage 2] Sinkhorn inference =====" | tee -a "$LOG_FILE"

cd $HGREC
python3 $REPO/scripts/task29x_stage2_codebook.py \
    --task_id $TASK_ID \
    --quantizer $QUANTIZER \
    --ckpt_path "$BEST_CKPT" \
    --output_path "$STAGE2_OUTPUT" \
    2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$STAGE2_OUTPUT" ]; then
    echo "❌ Stage 2 output MISSING" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Stage 2 done: $STAGE2_OUTPUT" | tee -a "$LOG_FILE"

# ========== Stage 3 T5-mini Training ==========
STAGE3_DIR=$REPO/products/task${TASK_ID}/stage3_t5mini
echo "" | tee -a "$LOG_FILE"
echo "===== [Task #${TASK_ID} Stage 3] T5-mini training (60m) =====" | tee -a "$LOG_FILE"

python3 $REPO/scripts/task29x_stage3_train.py \
    --task_id $TASK_ID \
    --code_path "$STAGE2_OUTPUT" \
    --out_dir "$STAGE3_DIR" \
    --epochs 60 \
    --batch_size 64 \
    --lr 1e-3 \
    2>&1 | tee -a "$LOG_FILE"

# 验证 Stage 3 ckpt
STAGE3_BEST=$(ls -t $STAGE3_DIR/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$STAGE3_BEST" ]; then
    echo "❌ Stage 3 HG_Rec_best.pth MISSING" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Stage 3 done: $STAGE3_BEST" | tee -a "$LOG_FILE"

# ========== Stage 4 R@10 Eval ==========
echo "" | tee -a "$LOG_FILE"
echo "===== [Task #${TASK_ID} Stage 4] Test R@10 evaluation =====" | tee -a "$LOG_FILE"

python3 $REPO/scripts/task29x_stage4_eval.py \
    --task_id $TASK_ID \
    --stage3_ckpt "$STAGE3_BEST" \
    --code_path "$STAGE2_OUTPUT" \
    --output_json $REPO/products/task${TASK_ID}/stage4_eval.json \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅✅✅ [Task #${TASK_ID} Stage 234] ALL DONE at $(date) =====" | tee -a "$LOG_FILE"
echo "result: Stage 4 R@10 = $(jq -r '.recall_at_10' $REPO/products/task${TASK_ID}/stage4_eval.json 2>/dev/null || echo 'CHECK JSON')" | tee -a "$LOG_FILE"