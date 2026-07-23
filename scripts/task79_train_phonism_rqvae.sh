#!/bin/bash
# Task #78 — phonism-style RQ-VAE brief training (5000 epochs) on Musical_Instruments sentence-t5-768d
# Architecture mirrors Task #126 phonism config but on Instruments (Toys checkpoint lost)
# input=768, e_dim=32, hidden=[512,256,128,64], codebook=256x3, sk_epsilons=[0,0,0.003]
set -e

GENE_REC=/home/wlia0047/ar57/wenyu/GeneRec
CKPT_DIR=$GENE_REC/products/task79_phonism_rqvae_768d/rqvae_ckpt
LOG=$GENE_REC/logs/task79_train_phonism_$(date +%j-%H%M%S).log

mkdir -p $CKPT_DIR

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd $GENE_REC/ETEGRec/RQVAE

CUDA_VISIBLE_DEVICES=2 \
TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task79 \
python -u main.py \
    --lr 1e-3 \
    --epochs 5000 \
    --batch_size 1024 \
    --weight_decay 1e-4 \
    --lr_scheduler_type linear \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.25 \
    --num_emb_list 256 256 256 \
    --sk_epsilons 0.0 0.0 0.003 \
    --layers 512 256 128 64 \
    --vq_type vq \
    --loss_type mse \
    --dist l2 \
    --device cuda:0 \
    --kmeans_init True \
    --data_path $GENE_REC/external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy \
    --ckpt_dir $CKPT_DIR \
    > $LOG 2>&1

echo "[DONE] Task #78 phonism training → $LOG"