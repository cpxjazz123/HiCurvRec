#!/bin/bash
# Parallel orchestrator: Task #156 (T5-small) GPU 3 + Task #157 (T5-base) GPU 0
# 2026-07-24 (用户 2026-07-24 21:07 选 C 方案)
#
# 阶段:
# 1. 等 Task #156 Stage 2 SID codebook 落盘 (在跑, 启动 21:07, ~3 min)
# 2. 并行 fire Task #156 Stage 3 (T5-small, GPU 3) + Task #157 Stage 3 (T5-base, GPU 0)
# 3. 等两个 Stage 3 都完成 (T5-base ~93h)
# 4. 并行 fire Task #156 Stage 4 (GPU 3) + Task #157 Stage 4 (GPU 0)
# 5. Done
#
# 该脚本 babysit 整个链 ~93h, nohup 后台跑, 退出 = 整链完成

set -eo pipefail

export PYTHONPATH=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec:${PYTHONPATH:-}

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/orchestrator
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/parallel_xy_$(date +%Y%m%d_%H%M%S).log
PID_FILE=/home/wlia0047/ar57/wenyu/GeneRec/products/orchestrator_pid

echo $BASHPID > "$PID_FILE"
echo "===== [Orchestrator] Parallel XY chain launched at $(date), PID=$BASHPID =====" | tee "$LOG_FILE"

CODE_FILE=/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_code_default.npy

# ========== Phase 1: 等 Stage 2 SID codebook 落盘 ==========
echo "[Phase 1] 等待 Stage 2 SID codebook: $CODE_FILE" | tee -a "$LOG_FILE"
WAIT_COUNT=0
while [ ! -f "$CODE_FILE" ]; do
    sleep 30
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 20)) -eq 0 ]; then
        echo "[Phase 1] 还在等 ($((WAIT_COUNT * 30))s)..." | tee -a "$LOG_FILE"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tee -a "$LOG_FILE"
    fi
    if [ $WAIT_COUNT -gt 600 ]; then
        echo "❌ Phase 1 等待超时 (5h)" | tee -a "$LOG_FILE"
        exit 1
    fi
done
SID_SIZE=$(stat -c '%s' "$CODE_FILE")
echo "✅ [Phase 1] SID codebook 落盘 ($SID_SIZE bytes), 启动 Stage 3 launches" | tee -a "$LOG_FILE"

# ========== Phase 2: 并行 Stage 3 ==========
echo "" | tee -a "$LOG_FILE"
echo "===== [Phase 2] 启动 Stage 3 (并行) =====" | tee -a "$LOG_FILE"

# Task #156 Stage 3 (T5-small, GPU 3)
T156_S3_LOG=/home/wlia0047/ar57/wenyu/GeneRec/logs/orchestrator/task156_stage3_$(date +%Y%m%d_%H%M%S).log
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task156_hgrec_stage3_train.sh > "$T156_S3_LOG" 2>&1 &
T156_PID=$!
echo "[Phase 2] Task #156 Stage 3 PID=$T156_PID, log=$T156_S3_LOG" | tee -a "$LOG_FILE"

sleep 5  # 间隔 5 sec 防止 CUDA init 冲突

# Task #157 Stage 3 (T5-base, GPU 0)
T157_S3_LOG=/home/wlia0047/ar57/wenyu/GeneRec/logs/orchestrator/task157_stage3_$(date +%Y%m%d_%H%M%S).log
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task157_hgrec_stage3_t5base_train.sh > "$T157_S3_LOG" 2>&1 &
T157_PID=$!
echo "[Phase 2] Task #157 Stage 3 PID=$T157_PID, log=$T157_S3_LOG" | tee -a "$LOG_FILE"

# 等两个都完成 (T5-base 估 ~93h)
echo "===== [Phase 2] 等 Stage 3 都完成... T156=$T156_PID, T157=$T157_PID =====" | tee -a "$LOG_FILE"
wait $T156_PID
T156_S3_EXIT=$?
echo "[Phase 2] Task #156 Stage 3 exit=$T156_S3_EXIT at $(date)" | tee -a "$LOG_FILE"

wait $T157_PID
T157_S3_EXIT=$?
echo "[Phase 2] Task #157 Stage 3 exit=$T157_S3_EXIT at $(date)" | tee -a "$LOG_FILE"

if [ $T156_S3_EXIT -ne 0 ] || [ $T157_S3_EXIT -ne 0 ]; then
    echo "❌ Stage 3 failed" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ [Phase 2] Stage 3 双双完成" | tee -a "$LOG_FILE"

# ========== Phase 3: 并行 Stage 4 eval ==========
echo "" | tee -a "$LOG_FILE"
echo "===== [Phase 3] 启动 Stage 4 (并行) =====" | tee -a "$LOG_FILE"

# Task #156 Stage 4 (GPU 3)
T156_S4_LOG=/home/wlia0047/ar57/wenyu/GeneRec/logs/orchestrator/task156_stage4_$(date +%Y%m%d_%H%M%S).log
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task156_hgrec_stage4_eval.sh > "$T156_S4_LOG" 2>&1 &
T156_S4_PID=$!
echo "[Phase 3] Task #156 Stage 4 PID=$T156_S4_PID" | tee -a "$LOG_FILE"

sleep 5

# Task #157 Stage 4 (GPU 0)
T157_S4_LOG=/home/wlia0047/ar57/wenyu/GeneRec/logs/orchestrator/task157_stage4_$(date +%Y%m%d_%H%M%S).log
nohup bash /home/wlia0047/ar57/wenyu/GeneRec/scripts/task157_hgrec_stage4_eval.sh > "$T157_S4_LOG" 2>&1 &
T157_S4_PID=$!
echo "[Phase 3] Task #157 Stage 4 PID=$T157_S4_PID" | tee -a "$LOG_FILE"

wait $T156_S4_PID
T156_S4_EXIT=$?
echo "[Phase 3] Task #156 Stage 4 exit=$T156_S4_EXIT at $(date)" | tee -a "$LOG_FILE"

wait $T157_S4_PID
T157_S4_EXIT=$?
echo "[Phase 3] Task #157 Stage 4 exit=$T157_S4_EXIT at $(date)" | tee -a "$LOG_FILE"

if [ $T156_S4_EXIT -eq 0 ] && [ $T157_S4_EXIT -eq 0 ]; then
    echo "✅ [Phase 3] Stage 4 双双完成. 整链 DONE" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Task #156 metrics: /home/wlia0047/ar57/wenyu/GeneRec/verdicts/task156_hgrec_code_default_metrics.json" | tee -a "$LOG_FILE"
    echo "Task #157 metrics: /home/wlia0047/ar57/wenyu/GeneRec/verdicts/task157_t5_base_metrics.json" | tee -a "$LOG_FILE"
else
    echo "❌ Phase 3 失败" | tee -a "$LOG_FILE"
    exit 1
fi

rm -f "$PID_FILE"
echo "===== [Orchestrator] DONE at $(date) =====" | tee -a "$LOG_FILE"
