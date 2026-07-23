#!/bin/bash
# Task #76 — ETEGRec 128d paper_exact 400 epoch 复跑 (baseline 重现)
# 完全复制 Task #73 paper_exact 配置: cycle=2, warmup=8000, warm_epoch=10, early_stop=15, 400 epoch
# GPU 2 绑定

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
RUN_NAME="task76_pe_${TS}"
LOG_FILE="$LOG_DIR/task76_etegrec_pe_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #76] ETEGRec 128d paper_exact replica launched at $(date) =====" | tee "$LOG_FILE"
echo "RUN_NAME: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Config: cycle=2, warmup=8000, warm_epoch=10, early_stop=15, epochs=400 (Task #73 replica)" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 main.py \
    --config ./config/musical_instruments_paper_exact.yaml \
    --lr_rec=0.005 \
    --lr_id=0.0001 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003 \
    --cycle=2 \
    --eval_step=2 \
    --early_stop=15 \
    --warmup_steps=8000 \
    --lr_scheduler_type=cosine \
    --code_length=4 \
    --batch_size=128 \
    --gradient_accumulation_steps=4 \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #76] ETEGRec 128d paper_exact replica completed at $(date) =====" | tee -a "$LOG_FILE"