#!/bin/bash
# Task #72 Phase 6: GPU 0 sequential baselines (now free after DECOR)
# Run remaining general recommenders + small sequential baselines

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole

LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs"

echo "===== [GPU 0] POST-DECOR launcher started at $(date) ====="

# Pop general (uni100 mode)
LOG_FILE="$LOG_DIR/task72_phase6_general_Pop_gpu0.log"
echo "===== [GPU 0] Starting Pop (general) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=Pop --gpu_id=0 --config=musical_instruments_general.yaml \
    >> "$LOG_FILE" 2>&1 || echo "===== [GPU 0] Pop (general) FAILED at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 0] Pop (general) done at $(date) =====" >> "$LOG_FILE"

# BPR general
LOG_FILE="$LOG_DIR/task72_phase6_general_BPR_gpu0.log"
echo "===== [GPU 0] Starting BPR (general) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=BPR --gpu_id=0 --config=musical_instruments_general.yaml \
    >> "$LOG_FILE" 2>&1 || echo "===== [GPU 0] BPR (general) FAILED at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 0] BPR (general) done at $(date) =====" >> "$LOG_FILE"

# ItemKNN general
LOG_FILE="$LOG_DIR/task72_phase6_general_ItemKNN_gpu0.log"
echo "===== [GPU 0] Starting ItemKNN (general) at $(date) =====" > "$LOG_FILE"
python3 run_recbole_baseline.py --model=ItemKNN --gpu_id=0 --config=musical_instruments_general.yaml \
    >> "$LOG_FILE" 2>&1 || echo "===== [GPU 0] ItemKNN (general) FAILED at $(date) =====" >> "$LOG_FILE"
echo "===== [GPU 0] ItemKNN (general) done at $(date) =====" >> "$LOG_FILE"

echo "===== [GPU 0] All 3 general baselines (Pop/BPR/ItemKNN) completed at $(date) ====="