#!/bin/bash
# Task #174 GPU 1 监控 daemon: 等 #169 Stage 3 early stop 后立即 launch #174 Stage 1.
# 监控 GPU 1 utilization, < 10% + memory < 5GB → 启动 #174 Stage 1.
# 监控 GPU 0 也 fall back (如 #170 先停).

set -e

LOG_FILE=/home/wlia0047/ar57/wenyu/GeneRec/logs/task174/_gpu_wait.log
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/logs/task174

echo "===== [Task #174 GPU wait daemon] started at $(date) =====" | tee -a $LOG_FILE

attempt=0
while true; do
    attempt=$((attempt+1))

    # 先尝试 GPU 1 (R7: 不抢占用卡)
    for gpu_id in 1 0 2 3; do
        line=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i $gpu_id 2>&1 | head -1)
        util=$(echo $line | cut -d',' -f2 | tr -d ' ' | sed 's/%//')
        mem_mib=$(echo $line | cut -d',' -f3 | tr -d ' ' | sed 's/MiB//')

        if [ -z "$util" ] || [ -z "$mem_mib" ]; then
            continue
        fi

        echo "[$(date)] attempt=$attempt gpu=$gpu_id util=${util}% mem=${mem_mib}MiB" >> $LOG_FILE

        if [ "$util" -lt 10 ] && [ "$mem_mib" -lt 5120 ]; then
            echo "✅ GPU $gpu_id free (util=${util}%, mem=${mem_mib}MiB). Launching #174 Stage 1" | tee -a $LOG_FILE
            CUDA_VISIBLE_DEVICES=$gpu_id bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task174_stage1_mckg_gating.sh
            exit 0
        fi
    done

    sleep 60
done