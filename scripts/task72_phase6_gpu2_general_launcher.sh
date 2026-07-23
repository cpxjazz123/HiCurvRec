#!/bin/bash
# Task #72 Phase 6: General recommenders on GPU 2 (FPMC re-run with neg sampling)
# FPMC needs NEG_ITEM_ID — use general yaml with train_neg_sample_args

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"

echo "===== [GPU 2] GENERAL baselines launcher started at $(date) ====="

# FPMC with neg sampling (general yaml)
LOG_FILE="$LOG_DIR/task72_phase6_general_FPMC_gpu2.log"
echo "===== [GPU 2] Starting FPMC (general) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=FPMC --gpu_id=2 --config=musical_instruments_general.yaml \
    >> "$LOG_FILE" 2>&1 || echo "===== [GPU 2] FPMC (general) FAILED at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 2] FPMC (general) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] GENERAL baselines completed at $(date) ====="