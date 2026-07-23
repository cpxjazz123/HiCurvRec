#!/bin/bash
# Task #72 Phase 6x: SINE sequential baseline on GPU 2 (idle after LightSANs done)
# SINE = Spatially-Induced Network (Cao et al. 2022)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] SINE launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_SINE_gpu2.log"
echo "===== [GPU 2] Starting SINE (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=SINE --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] SINE (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] SINE completed at $(date) ====="