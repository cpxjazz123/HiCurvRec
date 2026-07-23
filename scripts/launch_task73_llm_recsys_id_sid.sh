#!/bin/bash
# Task #73 — P5 SID only (CID already completed)
# After data.py fix for remapped_sequential branch
# After main.py line 310 dist.barrier() fix for non-distributed mode
set -eo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

# Critical: PYTHONPATH override so P5 uses transformers 4.x (not 5.x)
export PYTHONPATH=/home/wlia0047/ar57_scratch/wenyu/p5_libs:$PYTHONPATH

cd /home/wlia0047/arenyu/wenyu/GeneRec/external/LLM-RecSys-ID 2>/dev/null || cd /home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

DATA_DIR="/home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/"

LOG_FILE="$LOG_DIR/task73_llm_recsys_id_sid.log"
echo "===== [Task #73] SID restart @ $(date) =====" | tee "$LOG_FILE"
echo "data.py: remapped_sequential branch added (identity passthrough)" | tee -a "$LOG_FILE"

python -u main.py \
    --task instruments \
    --data_dir "$DATA_DIR" \
    --seed 2022 \
    --warmup_prop 0.05 \
    --lr 1e-3 \
    --clip 1.0 \
    --model_type 't5-small' \
    --epochs 20 \
    --gpu '0' \
    --logging_step 1000 \
    --logging_dir 'log/task73_instruments_SID.log' \
    --model_dir '/home/wlia0047/ar57/wenyu/GeneRec/products/task73/llm_recsys_id_sid.pt' \
    --train_sequential_item_batch 64 \
    --whole_word_embedding shijie \
    --item_representation remapped_sequential \
    --data_order remapped_sequential \
    --remapped_data_order original \
    --random_initialization_embedding \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #73] SID completed at $(date) =====" | tee -a "$LOG_FILE"