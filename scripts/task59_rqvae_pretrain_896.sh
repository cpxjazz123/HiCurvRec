#!/bin/bash
# Task #59 — RQ-VAE pretrain on 896d fused embedding (text 768 + SASRec 128)
# Output: 256-256-256-128_896din.rqvae.pth

set -eo pipefail

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/RQVAE

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE="$LOG_DIR/task59_rqvae_pretrain_${TS}.log"

mkdir -p "$LOG_DIR"

echo "===== [Task #59] RQ-VAE pretrain 896d started at $(date) =====" | tee "$LOG_FILE"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pretrain RQ-VAE on 896d fused embedding
# e_dim=128 (output dim before quantization), layers=[1024, 512, 256] (deeper for 896d input)
# num_emb_list=256x3 (3 hierarchical codes, 24 codes total per item)
python -u main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --e_dim 128 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 \
  --layers 1024 512 256 \
  --vq_type vq \
  --loss_type mse \
  --dist l2 \
  --device cuda:0 \
  --kmeans_init True \
  --data_path /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_896.npy \
  --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/RQVAE/rqvae_ckpt/Musical_Instruments_896din \
  2>&1 | tee -a "$LOG_FILE"

echo "===== [Task #59] RQ-VAE pretrain 896d completed at $(date) =====" | tee -a "$LOG_FILE"