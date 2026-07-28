#!/bin/bash
# Task #188 Phase 2: Generate 4 codebooks from 4 collision tiers (parallel on 4 GPUs)
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
mkdir -p $REPO/logs/task188 $REPO/HG-Rec/dataset/Instruments

# Tier 1: 8.6% → cuda:0
nohup python3 -u $REPO/scripts/task188_stage2_codebook.py \
    --ckpt_path $REPO/products/task188/codebook_t1_8pct/ckpt.pth \
    --output_path $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t1_8pct.npy \
    --device cuda:0 \
    > $REPO/logs/task188/stage2_t1_8pct.out 2>&1 &
PID1=$!
disown

# Tier 2: 10.1% → cuda:1
nohup python3 -u $REPO/scripts/task188_stage2_codebook.py \
    --ckpt_path $REPO/products/task188/codebook_t2_10pct/ckpt.pth \
    --output_path $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t2_10pct.npy \
    --device cuda:1 \
    > $REPO/logs/task188/stage2_t2_10pct.out 2>&1 &
PID2=$!
disown

# Tier 3: 11.2% → cuda:2
nohup python3 -u $REPO/scripts/task188_stage2_codebook.py \
    --ckpt_path $REPO/products/task188/codebook_t3_12pct/ckpt.pth \
    --output_path $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t3_12pct.npy \
    --device cuda:2 \
    > $REPO/logs/task188/stage2_t3_12pct.out 2>&1 &
PID3=$!
disown

# Tier 4: 12.8% → cuda:3
nohup python3 -u $REPO/scripts/task188_stage2_codebook.py \
    --ckpt_path $REPO/products/task188/codebook_t4_13pct/ckpt.pth \
    --output_path $REPO/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_t4_13pct.npy \
    --device cuda:3 \
    > $REPO/logs/task188/stage2_t4_13pct.out 2>&1 &
PID4=$!
disown

echo "$PID1 $PID2 $PID3 $PID4" > $REPO/products/task188/_STAGE2_PIDS
echo "Phase 2 launched: t1=$PID1 t2=$PID2 t3=$PID3 t4=$PID4"
