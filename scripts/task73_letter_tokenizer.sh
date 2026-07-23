#!/bin/bash
# Task #73 — LETTER tokenizer training on Musical_Instruments
# Uses sentence-t5-768d (text, 24588 items) + SASRec-32d CF embedding (PCA-projected from our 128d)
# Run on GPU 3

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/RQ-VAE

# Inputs (cached / pre-computed in Task #73 setup)
TEXT_EMB="/home/wlia0047/ar57/wenyu/GeneRec/external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy"
CF_EMB="/home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/RQ-VAE/ckpt/Instruments-sasrec32-task73.pt"
CKPT_DIR="/home/wlia0047/ar57/wenyu/GeneRec/products/task73/letter_tokenizer"
LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_FILE="$LOG_DIR/task73_letter_tokenizer.log"
echo "===== [Task #73] LETTER tokenizer started at $(date) =====" | tee "$LOG_FILE"

# LETTER hyperparameters from paper: alpha=0.01, beta=1e-4 (for Instruments)
# 4 codebooks 256 each, e_dim=32
python -u main.py \
    --device cuda:0 \
    --data_path "$TEXT_EMB" \
    --cf_emb "$CF_EMB" \
    --alpha 0.01 \
    --beta 0.0001 \
    --num_emb_list 256 256 256 256 \
    --e_dim 32 \
    --layers 1024 512 256 128 64 32 \
    --lr 1e-3 \
    --epochs 5000 \
    --batch_size 1024 \
    --weight_decay 1e-4 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 0.003 \
    --eval_step 50 \
    --learner Adam \
    --ckpt_dir "$CKPT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #73] LETTER tokenizer completed at $(date) =====" | tee -a "$LOG_FILE"
