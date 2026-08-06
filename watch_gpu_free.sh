#!/bin/bash
# Issue #64 GPU watcher — 等 4 GPU 全空 (bloger run_gr_rec_blo.py 结束) 自动重启 DDP.
# 检测逻辑: nvidia-smi 显示 4 GPU memused < 500MB 视为空闲, 然后启动 DDP.
# 检查间隔 30s, 超时 30 分钟退出.
set -uo pipefail

LOG=/tmp/v64_hab/watcher.log
LAUNCHER=/home/wlia0047/ar57/wenyu/GeneRec/launch_ddp_v64_hab.sh
MAX_WAIT_SEC=$((30 * 60))   # 30 min
INTERVAL=30
ELAPSED=0

echo "$(date '+%Y-%m-%d %H:%M:%S') watcher start, max wait ${MAX_WAIT_SEC}s" >> "$LOG"

while [ "$ELAPSED" -lt "$MAX_WAIT_SEC" ]; do
    # 找空闲 GPU 数量 (mem < 500MB 视为空)
    FREE_COUNT=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 | awk '$1 < 500 {n++} END {print n+0}')
    echo "$(date '+%Y-%m-%d %H:%M:%S') free_gpus=${FREE_COUNT}/4 (elapsed=$ELAPSED)" >> "$LOG"

    if [ "$FREE_COUNT" -ge 4 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') ALL 4 GPU FREE — launching DDP" >> "$LOG"
        # 清空产物
        rm -rf /tmp/v64_hab/* 2>/dev/null
        nohup bash "$LAUNCHER" > /tmp/v64_hab/ddp_launcher.log 2>&1 &
        LP=$!
        echo "$LP" > /tmp/v64_hab/_TRAINING_PID
        echo "$(date '+%Y-%m-%d %H:%M:%S') launched PID=$LP" >> "$LOG"
        exit 0
    fi

    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "$(date '+%Y-%m-%d %H:%M:%S') watcher TIMEOUT (${MAX_WAIT_SEC}s)" >> "$LOG"
exit 1