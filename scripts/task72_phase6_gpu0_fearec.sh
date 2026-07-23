#!/bin/bash
# Task #72 Phase 6y: FEARec sequential baseline on GPU 0 (idle after NeuMF done)
# FEARec = Frequency Enhanced Augmentation for Sequential Recommendation (Du et al. 2023)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 0] FEARec launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_FEARec_gpu0.log"
echo "===== [GPU 0] Starting FEARec (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=FEARec --gpu_id=0 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 0] FEARec (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 0] FEARec completed at $(date) ====="