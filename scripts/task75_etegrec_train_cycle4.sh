#!/bin/bash
# Task #75 — ETEGRec 128d cycle=4 (paper Section 4.1.5 备选 cycle 验证)
# 与 Task #74 paper-cycle 配置相同, 仅 cycle=4 vs cycle=2
# GPU 1 绑定

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
RUN_NAME="task75_c4_${TS}"
LOG_FILE="$LOG_DIR/task75_etegrec_c4_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #75] ETEGRec 128d cycle=4 launched at $(date) =====" | tee "$LOG_FILE"
echo "RUN_NAME: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Config: cycle=4, warmup=0, warm_epoch=1, early_stop=30 (paper-cycle)" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 main.py \
    --config ./config/musical_instruments_cycle4.yaml \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003 \
    --cycle=4 \
    --eval_step=2 \
    --early_stop=30 \
    --warmup_steps=0 \
    --lr_scheduler_type=cosine \
    --code_length=4 \
    --batch_size=128 \
    --gradient_accumulation_steps=4 \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #75] ETEGRec 128d cycle=4 completed at $(date) =====" | tee -a "$LOG_FILE"