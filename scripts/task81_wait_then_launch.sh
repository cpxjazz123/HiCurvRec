#!/bin/bash
# Task #81 — S3Rec launcher (waits for Task #80 FDSA to complete before starting)
set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/recbole_env

# Wait for Task #80 FDSA to finish (PID 4128724 — re-spawned at 16:58, 2026-07-23)
echo "[$(date +%H:%M:%S)] Waiting for FDSA (PID 4128724) to complete..."
while ps -p 4128724 > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] FDSA done. Freeing GPU 0..."

# Wait extra for GPU 0 to actually release
sleep 60

cd /home/wlia0047/ar57/wenyu/GeneRec/RecBole
GPU=0
LOG_FILE=/home/wlia0047/ar57/wenyu/GeneRec/logs/task81_s3rec_jul-23-2026_13-25-30.log
echo "===== Starting S3Rec on GPU $GPU at $(date) =====" | tee -a $LOG_FILE
python3 run_recbole_baseline.py --model=S3Rec --gpu_id=$GPU \
    --config=musical_instruments_sequential_paper.yaml \
    >> $LOG_FILE 2>&1 &
S3REC_PID=$!
# 写 PID 文件让 task88 v3 daemon 检测完成
echo $S3REC_PID > /home/wlia0047/ar57/wenyu/GeneRec/products/task81/_TRAINING_PID
echo "[$(date +%H:%M:%S)] S3Rec PID: $S3REC_PID" | tee -a $LOG_FILE
wait $S3REC_PID
echo "===== S3Rec completed at $(date) =====" | tee -a $LOG_FILE
rm -f /home/wlia0047/ar57/wenyu/GeneRec/products/task81/_TRAINING_PID
