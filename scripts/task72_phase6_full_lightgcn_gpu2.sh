#!/bin/bash
# Task #72 Phase 6: LightGCN on GPU 2 (Random finished)
# Mode: full ranking, single model

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments_general.yaml"

echo "===== [GPU 2] Starting LightGCN (full) at $(date) =====" > "$LOG_DIR/task72_phase6_full_LightGCN_gpu2.log"
python3 run_recbole_baseline.py --model=LightGCN --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_DIR/task72_phase6_full_LightGCN_gpu2.log" 2>&1
echo "===== [GPU 2] LightGCN (full) completed at $(date) =====" >> "$LOG_DIR/task72_phase6_full_LightGCN_gpu2.log"
echo "===== [GPU 2] LightGCN done at $(date) ====="
