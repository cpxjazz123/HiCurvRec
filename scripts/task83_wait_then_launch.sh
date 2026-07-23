#!/bin/bash
# Task #83 — P5-SID wait launcher (等 task82 P5-CID 完成)
set -e
T82_PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/task82/_TRAINING_PID
echo "[$(date +%H:%M:%S)] Waiting for task82 P5-CID..."
while [ ! -f $T82_PID_FILE ] || kill -0 $(cat $T82_PID_FILE) 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] task82 done. Starting task83 P5-SID..."
bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task83_p5_sid_instruments.sh
