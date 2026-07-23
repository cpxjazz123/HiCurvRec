#!/bin/bash
# Task #72 Phase 6: Extended baselines on GPU 2 (FPMC -> HGN)
# GPU 2 freed after STAMP completion (00:41)

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] EXTENDED full-ranking launcher started at $(date) ====="
echo "===== Config: $CONFIG, mode: full, PHYSICAL gpu_id=2 ====="

# FPMC (sequential, smaller than Caser ~10 min/epoch)
LOG_FILE="$LOG_DIR/task72_phase6_full_FPMC_gpu2.log"
echo "===== [GPU 2] Starting FPMC (full) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=FPMC --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1 || echo "===== [GPU 2] FPMC (full) FAILED at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] FPMC done at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] FPMC done at $(date) ====="

# HGN
LOG_FILE="$LOG_DIR/task72_phase6_full_HGN_gpu2.log"
echo "===== [GPU 2] Starting HGN (full) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=HGN --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1 || echo "===== [GPU 2] HGN (full) FAILED at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] HGN done at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] HGN done at $(date) ====="

echo "===== [GPU 2] EXTENDED baselines (FPMC/HGN) completed at $(date) ====="