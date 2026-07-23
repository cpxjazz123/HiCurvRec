#!/bin/bash
# Task #72 Phase 6w: LightSANs sequential baseline on GPU 2 (idle after GCSAN done)
# LightSANs = Lightweight Self-Attentive Network for recommendation

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"
CONFIG="musical_instruments.yaml"

echo "===== [GPU 2] LightSANs launcher started at $(date) ====="

LOG_FILE="$LOG_DIR/task72_phase6_full_LightSANs_gpu2.log"
echo "===== [GPU 2] Starting LightSANs (full mode) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=LightSANs --gpu_id=2 --config="$CONFIG" \
    >> "$LOG_FILE" 2>&1
echo "===== [GPU 2] LightSANs (full) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 2] LightSANs completed at $(date) ====="
