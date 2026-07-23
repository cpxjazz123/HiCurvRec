#!/bin/bash
# Task #78 — TIGER T5 训练 (paper Table 2 baseline #9, R@10 paper=0.0574)
# 关键路径修复 (2026-07-23):
#   - index_file: 用绝对路径 (绕过 data.py:141 错误拼接逻辑 dataset+index_file)
#   - sid 转换: task78_sid_pt_to_json_letter.py → Instruments_tiger.index.json (0 collision)
# GPU 0 单卡 (Task #77 经验: 必须独占)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task78_tiger_t5_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #78] TIGER T5 training launched at $(date) =====" | tee "$LOG_FILE"
echo "Backbone: t5-small (TIGER paper default)" | tee -a "$LOG_FILE"
echo "index_file: ABS PATH /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER/Instruments_tiger.index.json (task78 sid pt→json, 24588×4 tokens, 0 collision)" | tee -a "$LOG_FILE"
echo "data_path: /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data" | tee -a "$LOG_FILE"
echo "Target: R@10 ≥ 0.0574 (paper Musical_Instruments TIGER)" | tee -a "$LOG_FILE"

# GPU 0 单卡
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=0

DATASET=Instruments
OUTPUT_DIR=./ckpt/${DATASET}_tiger/
mkdir -p "$OUTPUT_DIR"

# 关键: --index_file 用绝对路径, 绕过 data.py:141 dataset+filename 拼接 bug
INDEX_FILE="/home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER/Instruments_tiger.index.json"
[ -f "$INDEX_FILE" ] || { echo "[FATAL] $INDEX_FILE 不存在"; exit 1; }

python3 finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --base_model t5-small \
    --per_device_batch_size 128 \
    --learning_rate 5e-4 \
    --epochs 100 \
    --index_file "$INDEX_FILE" \
    --temperature 1.0 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #78] TIGER T5 training completed at $(date) =====" | tee -a "$LOG_FILE"
