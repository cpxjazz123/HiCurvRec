#!/bin/bash
set -uo pipefail
REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO
LOG_DIR=$REPO/logs/task312
mkdir -p "$LOG_DIR"
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/stage2_codebook_${TS}.log
echo "[$(date)] Task #312 Stage 2 launched"
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN $REPO/scripts/task312_issue35_rl_identity_sl22_stage2_codebook.py > "$LOG_FILE" 2>&1
echo "[$(date)] Stage 2 done, exit=$?"
