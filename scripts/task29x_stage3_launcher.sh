#!/bin/bash
# Task #290/#291/#292 — Stage 3 T5-mini training (60 epochs) + Stage 4 eval
# 2026-07-29: 3 个 Stage 2 已完成 (SID npy 落盘). 立即启动 Stage 3 + Stage 4.
#
# 用法: bash task29x_stage3_launcher.sh <task_id> <gpu_id>
#   task_id: 290/291/292
#   gpu_id: 0/1/2/3 (R7 不抢已占卡)

set -e

TASK_ID=${1:-290}
GPU_ID=${2:-0}
REPO=/home/wlia0047/ar57/wenyu/GeneRec
HGREC=$REPO/HG-Rec

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# 确定 quantizer type from task_id
case $TASK_ID in
  290) QUANTIZER=fsq; NPY_SUFFIX=fsk ;;
  291) QUANTIZER=ema; NPY_SUFFIX=ema ;;
  292) QUANTIZER=restoration; NPY_SUFFIX=restoration ;;
  *) echo "❌ Unknown task_id $TASK_ID"; exit 1 ;;
esac

LOG_DIR=$REPO/logs/task${TASK_ID}
mkdir -p $LOG_DIR $REPO/products/task${TASK_ID}
TS=$(date +%d-%H-%M-%S)
LOG_FILE=$LOG_DIR/stage34_${TS}.log

echo "===== [Task #${TASK_ID} Stage 34] QUANTIZER=${QUANTIZER} GPU=${GPU_ID} at $(date) =====" | tee "$LOG_FILE"

# GPU + Triton cache (R7 + L40S sm_89 必须专属 cache dir, 否则会复用旧 A40 sm_86 编译产物)
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task${TASK_ID}
mkdir -p $TRITON_CACHE_DIR
# PYTHONPATH for `from data.X` / `from model.X` (R-fix: 之前漏了这个导致 ModuleNotFoundError)
export PYTHONPATH=$HGREC:$PYTHONPATH

# Stage 2 SID npy
SID_NPY=$HGREC/dataset/Instruments/Instruments_t5_hrqvae_${NPY_SUFFIX}.npy
if [ ! -f "$SID_NPY" ]; then
    echo "❌ Stage 2 SID npy MISSING: $SID_NPY" | tee -a "$LOG_FILE"
    exit 1
fi
echo "Stage 2 SID: $SID_NPY" | tee -a "$LOG_FILE"

# ========== Stage 3 T5-mini Training ==========
STAGE3_DIR=$REPO/products/task${TASK_ID}/stage3_t5mini
echo "" | tee -a "$LOG_FILE"
echo "===== [Task #${TASK_ID} Stage 3] T5-mini training =====" | tee -a "$LOG_FILE"

cd $REPO
python3 scripts/task84_hgrec_stage3_train.py \
    --dataset_path $HGREC/dataset/ \
    --dataset_name Instruments \
    --code_path _t5_hrqvae_${NPY_SUFFIX}.npy \
    --codebook_size 64 128 256 1 \
    --log_path $REPO/logs/task${TASK_ID}/stage3/ \
    --save_path $STAGE3_DIR/ \
    --num_epochs 30 \
    --batch_size 256 \
    --lr 1e-3 \
    --beam_size 20 \
    --topk_list 5 10 20 \
    --seed 42 \
    --disable_early_stop \
    2>&1 | tee -a "$LOG_FILE"

STAGE3_BEST=$(ls -t $STAGE3_DIR/Instruments/*/HG_Rec_best.pth 2>/dev/null | head -1)
if [ -z "$STAGE3_BEST" ] || [ ! -f "$STAGE3_BEST" ]; then
    echo "❌ Stage 3 HG_Rec_best.pth MISSING" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Stage 3 done: $STAGE3_BEST" | tee -a "$LOG_FILE"

# ========== Stage 4 R@10 Eval ==========
echo "" | tee -a "$LOG_FILE"
echo "===== [Task #${TASK_ID} Stage 4] Test R@10 evaluation =====" | tee -a "$LOG_FILE"

cd $REPO
python3 scripts/task29x_stage4_eval.py \
    --task_id $TASK_ID \
    --stage3_ckpt "$STAGE3_BEST" \
    --code_path "$SID_NPY" \
    --dataset_path $HGREC/dataset/ \
    --dataset_name Instruments \
    --output_json $REPO/products/task${TASK_ID}/stage4_eval.json \
    --device cuda:0 \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅✅✅ [Task #${TASK_ID} Stage 34] ALL DONE at $(date) =====" | tee -a "$LOG_FILE"
RECALL10=$(jq -r '.test_recalls["Recall@10"] // empty' $REPO/products/task${TASK_ID}/stage4_eval.json 2>/dev/null || echo "JSON_MISSING")
echo "result: Stage 4 R@10 = $RECALL10 (vs HG-Rec baseline 0.1020)" | tee -a "$LOG_FILE"