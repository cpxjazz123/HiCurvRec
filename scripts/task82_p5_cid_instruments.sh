#!/bin/bash
# Task #82 — P5-CID 训练 (paper Table 2 baseline #12, R@10 paper=0.0507)
# 基于 LLM-RecSys-ID Wenyueh 原作者 repo + instruments CID 索引
# 不走 P5 OpenP5 (transformers 5.x 不兼容), 直接用 LLM-RecSys-ID 已验证的 main.py
# GPU 1 单卡 (vs Task #78 GPU 0 TIGER 并行, R7 单卡独占)
#
# 模板来源: task73_instruments_CID.log (已跑过 53K steps, 但被用户 kill; 重启)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/llm_recsys_id_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task82_p5_cid_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #82] P5-CID training launched at $(date) =====" | tee "$LOG_FILE"
echo "Repo: LLM-RecSys-ID (Wenyueh/LLM-RecSys-ID, paper arxiv:2305.06569)" | tee -a "$LOG_FILE"
echo "item_representation=CF (collaborative indexing), cluster=20×500 (24588 items)" | tee -a "$LOG_FILE"
echo "Target: R@10 ≥ 0.0507 (paper Musical_Instruments P5-CID)" | tee -a "$LOG_FILE"

# GPU 1 (Task #78 占 GPU 0, R7 强制单卡独占)
export CUDA_VISIBLE_DEVICES=1

python3 main.py \
    --seed 42 \
    --data_dir /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/ \
    --logging_dir /home/wlia0047/ar57/wenyu/GeneRec/logs/task82_p5_cid_${TS}.log \
    --model_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task82/p5_cid_instruments.pt \
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
# R12 新规则: 每个训练必须在固定阶段 (epoch 末) 保存 checkpoint. 写 PID 文件让 daemon 检测
echo $TRAIN_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/task82/_TRAINING_PID
echo "[$(date +%H:%M:%S)] task82 training PID: $TRAIN_PID" | tee -a "$LOG_FILE"
wait $TRAIN_PID
echo "===== [Task #82] P5-CID training completed at $(date) =====" | tee -a "$LOG_FILE"
rm -f /home/wlia0047/ar57/wenyu/GeneRec/products/task82/_TRAINING_PID
