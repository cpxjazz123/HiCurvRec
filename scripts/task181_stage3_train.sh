#!/bin/bash
# Task #181 Stage 3 — HG-Rec T5-small training (100% official aligned Phase 0.6 baseline)
# Uses Stage 2 codebook output: Instruments_t5_rqvae_paper_fix.npy
# GPU 1 (same as Stage 1, Stage 2 is done)
# R12: save_limit=1 (save new ckpt 时删旧)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_phase0.6
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task181

# Verify Stage 2 codebook exists
CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_paper_fix.npy
if [ ! -f "$CODE_FILE" ]; then
    echo "❌ $CODE_FILE NOT FOUND — Stage 2 not complete"
    exit 1
fi
echo "✅ Stage 2 codebook found: $CODE_FILE"

echo "===== [Task #181 Stage 3] HG-Rec T5-small training at $(date) ====="

python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
    --code_path _t5_rqvae_paper_fix.npy \
    --codebook_size 64 128 256 1 \
    --num_epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:1 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_phase0.6/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 20 \
    --beam_size 20 \
    --infer_size 96 \
    2>&1 | tee /home/wlia0047/ar57/wenyu/GeneRec/logs/task181/stage3_train.out

echo "===== [Task #181 Stage 3] completed at $(date) ====="
