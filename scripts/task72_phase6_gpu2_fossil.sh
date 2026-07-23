#!/bin/bash
# Task #72 Phase 6t: FOSSIL sequential baseline on GPU 2 (idle after DIN fail)
# FOSSIL = Fusing Similarity Models with Markov Chains (He et al. 2016)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] FOSSIL launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_FOSSIL_gpu2.log"
echo "===== [GPU 2] Starting FOSSIL (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=FOSSIL --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] FOSSIL (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] FOSSIL completed at $(date) ====="
