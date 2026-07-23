#!/bin/bash
# Task #86 — P5-SID standalone eval (paper Table 2 baseline #13)
# 基于 LLM-RecSys-ID --eval_only mode 加载 pre-trained ckpt
# GPU 1 (R7 空闲, P5-SID 训练占 GPU 2)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/llm_recsys_id_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task86_p5_sid_eval_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #86] P5-SID standalone eval launched at $(date) =====" | tee "$LOG_FILE"
echo "Repo: LLM-RecSys-ID (Wenyueh/LLM-RecSys-ID, paper arxiv:2305.06569)" | tee -a "$LOG_FILE"
echo "item_representation=remapped_sequential (sequential indexing)" | tee -a "$LOG_FILE"
echo "Target: R@10 ≥ 0.0234 (paper Musical_Instruments P5-SID)" | tee -a "$LOG_FILE"

# GPU 1 (R7 强制空闲, P5-SID 训练占 GPU 2, S3Rec 训练占 GPU 0)
export CUDA_VISIBLE_DEVICES=1

python3 main.py \
    --eval_only \
    --seed 42 \
    --data_dir /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/ \
    --logging_dir "$LOG_FILE" \
    --model_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task83/p5_sid_instruments.pt \
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
    --epochs 10 \
    --lr 1e-3 \
    --clip 1 \
    --logging_step 1000 \
    --warmup_prop 0.05 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.01 \
    --adam_eps 1e-6 \
    --dropout 0.1 \
    --alpha 2 \
    --multiGPU \
    --gpu 1 \
    --evaluation_method sequential_item \
    --evaluation_template_id 0 \
    --number_of_items 24588 \
    --item_representation remapped_sequential \
    --cluster_number 55 \
    --cluster_size 100 \
    --data_order remapped_sequential \
    --remapped_data_order original \
    --resolution 2 \
    --max_random_number 30000 \
    --min_random_number 1000 \
    --random_initialization_embedding \
    --whole_word_embedding shijie \
    2>&1 | tee -a "$LOG_FILE" &

EVAL_PID=$!
echo "[$(date +%H:%M:%S)] task86 P5-SID eval PID: $EVAL_PID" | tee -a "$LOG_FILE"
wait $EVAL_PID
echo "===== [Task #86] P5-SID eval completed at $(date) =====" | tee -a "$LOG_FILE"
