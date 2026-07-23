#!/bin/bash
# Task #73 ETEGRec paper-EXACT rerun (R@10 target ≥ 0.0586, paper 0.0624)
# Diff vs prior paper_aligned.sh:
#   (1) dec_cl_loss: 1e-4 → 3e-4 (paper exact)
#   (2) lr_rec: 3e-3 → 5e-3 (paper exact)
#   (3) Add --warmup_steps=8000 --lr_scheduler_type=cosine (paper default)
#   (4) --epochs=400 (paper exact, was early_stop 15)
#   (5) Use pretrained RQ-VAE 256-256-256-128.rqvae.pth (yaml default, was bypassed before)
#   (6) --code_length=4 (paper, was 3)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
RUN_NAME="task73_pe_${TS}"
LOG_FILE="$LOG_DIR/task73_etegrec_pe_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #73] ETEGRec paper-EXACT launched at $(date) =====" | tee "$LOG_FILE"
echo "RUN_NAME: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Paper-exact config: lr_rec=5e-3, dec_cl=3e-4, warmup=8000, cosine, pretrained RQ-VAE, code_length=4, epochs=400" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# batch_size 512 OOMs on 44GB L40S — use bs=128 + grad_accum=4 (effective 512)
# early_stop=15 per yaml (paper)
python3 main.py \
    --config ./config/musical_instruments.yaml \
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

echo "===== [Task #73] ETEGRec paper-EXACT completed at $(date) =====" | tee -a "$LOG_FILE"
echo "===== [Task #73] ETEGRec paper-EXACT done at $(date) ====="