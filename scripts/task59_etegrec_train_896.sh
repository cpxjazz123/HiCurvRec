#!/bin/bash
# Task #59 — ETEGRec 896d fused emb retraining
# Uses 896d fused (text 768 + SASRec 128) embedding + new RQ-VAE ckpt
# Diff vs paper_exact.sh:
#   - semantic_emb_path: 128d → 896d (text 768 + SASRec 128)
#   - layers: [256,128] → [1024,512,256] (deeper for 896d input)
#   - RQ-VAE ckpt: 256-256-256-128.rqvae.pth → 256-256-256-128_896din.rqvae.pth
# Goal: R@10 ≥ 0.040 (target -62% → reduce gap by 50%)

set -eo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
RUN_NAME="task59_etegrec_896_${TS}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #59] ETEGRec 896d launched at $(date) =====" | tee "$LOG_FILE"
echo "RUN_NAME: $RUN_NAME" | tee -a "$LOG_FILE"
echo "Config: 896d fused emb (text 768 + SASRec 128), layers=[1024,512,256], pretrained RQ-VAE collision=9.33%" | tee -a "$LOG_FILE"

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

echo "===== [Task #59] ETEGRec 896d completed at $(date) =====" | tee -a "$LOG_FILE"