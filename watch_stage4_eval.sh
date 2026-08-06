#!/bin/bash
# Issue #64 Stage4 watcher — 训练 ckpt 一旦落盘 (/tmp/v64_hab/HG_Rec_best.pth) 自动触发 Stage4 eval.
# Stage4 eval 路径: common/stage4/stage4_eval_pure_t5.py
set -uo pipefail

LOG=/tmp/v64_hab/stage4_watcher.log
CKPT=/tmp/v64_hab/HG_Rec_best.pth
EVAL_OUT=/tmp/v64_hab/stage4_eval.log
EVAL_PID_FILE=/tmp/v64_hab/_EVAL_PID
STAGE4=/home/wlia0047/ar57/wenyu/GeneRec/common/stage4/stage4_eval_pure_t5.py
GENREC_PY=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3

INTERVAL=30
ELAPSED=0
MAX_WAIT=$((90 * 60))   # 90 min (训练 20min + 余量)

echo "$(date '+%Y-%m-%d %H:%M:%S') stage4 watcher start" >> "$LOG"

while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    if [ -f "$CKPT" ]; then
        SIZE=$(stat -c%s "$CKPT" 2>/dev/null)
        # ckpt 完整大小 ~22MB (model state_dict), >5MB 视为完整
        if [ "$SIZE" -gt 5000000 ]; then
            # 等训练结束 (launcher 死 = 训练完; 或 ckpt mtime > 5 min)
            CKPT_AGE=$(( $(date +%s) - $(stat -c%Y "$CKPT") ))
            # 等 DDP launcher 死 (即 stage3 训练已结束)
            DDP_LAUNCHER_PID=$(cat /tmp/v64_hab/_TRAINING_PID 2>/dev/null || echo "")
            DDP_ALIVE="no"
            if [ -n "$DDP_LAUNCHER_PID" ] && kill -0 "$DDP_LAUNCHER_PID" 2>/dev/null; then
                DDP_ALIVE="yes"
            fi
            echo "$(date '+%Y-%m-%d %H:%M:%S') ckpt found size=$SIZE age=${CKPT_AGE}s ddp_alive=$DDP_ALIVE" >> "$LOG"

            if [ "$DDP_ALIVE" = "no" ] && [ "$CKPT_AGE" -gt 10 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') launching Stage4 eval" >> "$LOG"
                nohup "$GENREC_PY" -u "$STAGE4" \
                    --ckpt_path "$CKPT" \
                    --sid_npy /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/sid_output.npy \
                    --product_dir /tmp/v64_hab \
                    --tag taskA_issue64_hab_ddp \
                    --hyperbolic_attn_bias \
                    --hab_lambda_max 0.20 \
                    --hab_stage2_ckpt /home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/taskA_stage2_issue61/taskA_stage2_issue61_ckpt.pth > "$EVAL_OUT" 2>&1 &
                EP=$!
                echo "$EP" > "$EVAL_PID_FILE"
                echo "$(date '+%Y-%m-%d %H:%M:%S') stage4 launched PID=$EP" >> "$LOG"
                exit 0
            fi
        fi
    fi
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "$(date '+%Y-%m-%d %H:%M:%S') stage4 watcher TIMEOUT" >> "$LOG"
exit 1