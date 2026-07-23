#!/bin/bash
# Task #72 Phase 4: ETEGRec RQVAE pretraining for Musical_Instruments
# Uses 1 GPU. Data must be prepped first by task72_phase4_etegrec_prep_data.py

set -e  # exit on error

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/RQVAE

# Use GPU passed as arg (default: cuda:0)
GPU="${1:-cuda:0}"
echo "Running on GPU: $GPU"
echo "Starting at: $(date)"

python main.py \
    --lr 1e-3 \
    --epochs 10000 \
    --batch_size 1024 \
    --e_dim 128 \
    --beta 0.25 \
    --num_emb_list 256 256 256 \
    --layers 256 128 \
    --vq_type vq \
    --loss_type mse \
    --dist l2 \
    --data_path ../dataset/Musical_Instruments/Musical_Instruments_emb_128.npy \
    --ckpt_dir ../dataset/Musical_Instruments/ \
    --device "$GPU" \
    --num_workers 4

echo "RQVAE training completed at: $(date)"

# Find and copy best model to expected config path
CKPT_DIR=$(ls -td ../dataset/Musical_Instruments/2*/ | head -1)
if [ -n "$CKPT_DIR" ]; then
    BEST_MODEL="${CKPT_DIR}best_loss_model.pth"
    if [ -f "$BEST_MODEL" ]; then
        cp "$BEST_MODEL" ../dataset/Musical_Instruments/256-256-256-128.rqvae.pth
        echo "Copied best model to ../dataset/Musical_Instruments/256-256-256-128.rqvae.pth"
    else
        echo "Warning: best_loss_model.pth not found in $CKPT_DIR"
        echo "Checkpoint contents:"
        ls -la "${CKPT_DIR}"
    fi
else
    echo "Warning: No timestamped checkpoint directory found"
fi
