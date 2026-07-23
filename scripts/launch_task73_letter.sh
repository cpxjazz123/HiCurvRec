#!/bin/bash
# Task #73 — LETTER RQ-VAE training (substitute emb + SASRec 32-d cf_emb)
# Substitute LLaMA-TD embedding (missing) with BGE-base-en-v1.5 768-d
# cf_emb: SASRec hidden_size=32 (L2 normalized)
# Run on GPU 1 (ETEGRec GPU 0, P5 GPU 2)
set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_FILE="$LOG_DIR/task73_letter_rqvae.log"
echo "===== [Task #73] LETTER RQ-VAE started at $(date) =====" | tee "$LOG_FILE"
echo "emb: BGE-base-en-v1.5 768-d (substitute for LLaMA-TD 4096-d)" | tee -a "$LOG_FILE"
echo "cf_emb: SASRec hidden_size=32 (paper LETTER default)" | tee -a "$LOG_FILE"

# alpha=0.01 beta=0.0001 (from LETTER paper)
# RQ-VAE config: 4 codebooks (num_emb_list=[256,256,256,256])
$CONDA_PREFIX/bin/python3 -u RQ-VAE/main.py \
    --device cuda:0 \
    --data_path ./data/Instruments/Instruments.emb-bge.npy \
    --alpha 0.01 \
    --beta 0.0001 \
    --cf_emb ./RQ-VAE/ckpt/Instruments-sasrec32-v2.pt \
    --ckpt_dir ../checkpoint/Instruments_bge_sasrec32/ \
    --lr 1e-3 \
    --epochs 5000 \
    --batch_size 1024 \
    --num_emb_list 256 256 256 256 \
    --e_dim 32 \
    --layers 2048 1024 512 256 128 64 \
    2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #73] LETTER RQ-VAE completed at $(date) =====" | tee -a "$LOG_FILE"