#!/bin/bash
# Task #151 — P5-CID 5-任务联合训练 (paper-aligned multi-task, after main.py patch)
# 2026-07-24
#
# 根因: LLM-RecSys-ID main.py:192 number_of_tasks=1, 训练循环只 iterate train_loaders[0]
# 5 个 P5 task 中只用了 1 个 (sequential_item), 其它 4 个 loaded but never trained.
# 修复: scripts/task151_patch_main.py dry-run + apply (User OK per "帮我修复- P5-CID")
#
# R7 GPU: GPU 0 空闲 (Task #149 stage 3 占 GPU 1, Task #150 LETTER 占 GPU 2+3)
# 单 GPU 模式: --multiGPU 移除, --gpu 0, 节省 GPU 占用 + 不抢其它任务

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task151

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task151
mkdir -p "$LOG_DIR"

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID

TRAIN_PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task151/_TRAINING_PID
LOG_FILE="$LOG_DIR/p5_cid_5task_$(date +%Y-%m-%d_%H-%M-%S).log"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/llm_recsys_id_env

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task151

echo "[$(date)] Task #151 P5-CID 5-task launcher (patched main.py — cycles all 5 dataloaders)"

python3 main.py \
    --seed 42 \
    --data_dir /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/ \
    --logging_dir "$LOG_FILE" \
    --model_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task151/p5_cid_5task.pt \
    --task instruments \
    --max_history 20 \
    --sequential_num 10 \
    --negative_sample 2 \
    --yes_no_sample 5 \
    --direct_item_proportion 2 \
    --train_sequential_item_batch 64 \
    --train_sequential_yesno_batch 32 \
    --train_direct_yesno_batch 48 \
    --train_direct_candidate_batch 12 \
    --train_direct_straightforward_batch 48 \
    --meta_epochs 10 \
    --meta_lr 1e-4 \
    --review_epochs 2 \
    --review_lr 1e-4 \
    --model_type t5-small \
    --epochs 2 \
    --lr 1e-3 \
    --clip 1 \
    --logging_step 1000 \
    --warmup_prop 0.05 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.01 \
    --adam_eps 1e-6 \
    --dropout 0.1 \
    --alpha 2 \
    --gpu 0 \
    --evaluation_method sequential_item \
    --evaluation_template_id 0 \
    --number_of_items 24588 \
    --item_representation CF \
    --cluster_number 20 \
    --cluster_size 500 \
    --data_order remapped_sequential \
    --remapped_data_order original \
    --resolution 2 \
    --max_random_number 30000 \
    --min_random_number 1000 \
    --random_initialization_embedding \
    --whole_word_embedding shijie \
    2>&1 | tee -a "$LOG_FILE" &

TRAIN_PID=$!
echo $TRAIN_PID > "$TRAIN_PID_FILE"
echo "[$(date)] task151 training PID: $TRAIN_PID, log: $LOG_FILE"
wait $TRAIN_PID
EXIT_CODE=$?
rm -f "$TRAIN_PID_FILE"
echo "[$(date)] task151 training done, exit=$EXIT_CODE"
