#!/bin/bash
# Task #211 Phase 1 — 提前中止监控 (per user 2026-07-26)
# 监控 epoch 20 + 50 检查两个 gate:
#   1. hyp_norm_max > 0.98 → "boundary saturation" (B1 失效模式)
#   2. dyn_range < 1.5 → "distance saturation" (A3 失效模式)
# 任何 epoch 触发 → kill 对应 PID + 写 abort 日志
# R11.4 critical decision: 修改上游 train_hrqvae.py 加 abort hook 风险大, 改用 shell 监控.

REPO=/home/wlia0047/ar57/wenyu/GeneRec
LOG_DIR=$REPO/logs/task211
PID_DIR=$REPO/products/task211

# arm_name → log_file
declare -A ARM_LOGS=(
    [C1]=$LOG_DIR/C1_stage1_train.out
    [C2]=$LOG_DIR/C2_stage1_train.out
    [C3]=$LOG_DIR/C3_stage1_train.out
)

# arm_name → pid_file
declare -A ARM_PIDS=(
    [C1]=$PID_DIR/_TRAINING_PID_C1
    [C2]=$PID_DIR/_TRAINING_PID_C2
    [C3]=$PID_DIR/_TRAINING_PID_C3
)

CHECK_EPOCHS=(20 50)
ABORT_FLAG_DIR=$PID_DIR/_aborted

mkdir -p $ABORT_FLAG_DIR

# 每 60 秒检查一次, 持续 2 小时
for i in $(seq 1 120); do
    sleep 60
    for arm in C1 C2 C3; do
        log=${ARM_LOGS[$arm]}
        pid_file=${ARM_PIDS[$arm]}

        # 检查是否已 abort
        if [ -f "$ABORT_FLAG_DIR/$arm" ]; then
            continue
        fi

        # 检查训练是否还在跑
        if [ ! -f "$pid_file" ]; then
            continue
        fi
        pid=$(cat "$pid_file")
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$(date)] $arm: PID $pid 已停, 跳过"
            continue
        fi

        # 读最新 epoch
        cur_epoch=$(grep -oE "epoch [0-9]+ training" "$log" 2>/dev/null | tail -1 | awk '{print $2}')
        if [ -z "$cur_epoch" ]; then
            continue  # 还没开始第一个 epoch
        fi

        # 检查是否到 check epoch
        for check_ep in "${CHECK_EPOCHS[@]}"; do
            # 当 epoch 跨越 check_ep+5 时触发检查 (给一个缓冲)
            if [ "$cur_epoch" -ge "$check_ep" ] && [ "$cur_epoch" -le $((check_ep + 5)) ]; then
                # 检查 gate: hypnorm max ‖x‖_E > 0.98
                # log 格式: "L0 κ=1.000 ‖x‖_E=[0.050,0.062,0.082] mean=0.064 ..."
                # 取 L0 max (第一个元素)
                hypnorm_line=$(grep "hypnorm" "$log" 2>/dev/null | tail -1)
                if [ -n "$hypnorm_line" ]; then
                    # 提取 L0 ‖x‖_E= 后面第一个数 (min 值, 0.050 在 [0.050,0.062,0.082] 是 min)
                    # 实际我们想 max, 即第三个值. 用 awk 解析 [a,b,c]
                    l0_max=$(echo "$hypnorm_line" | grep -oE 'L0 κ=[0-9.]+ ‖x‖_E=\[[0-9.,]+\]' | head -1 | awk -F'[][,]' '{print $5}')
                    l1_max=$(echo "$hypnorm_line" | grep -oE 'L1 κ=[0-9.]+ ‖x‖_E=\[[0-9.,]+\]' | head -1 | awk -F'[][,]' '{print $5}')
                    l2_max=$(echo "$hypnorm_line" | grep -oE 'L2 κ=[0-9.]+ ‖x‖_E=\[[0-9.,]+\]' | head -1 | awk -F'[][,]' '{print $5}')
                    overall_max=$(printf "%s\n%s\n%s\n" "$l0_max" "$l1_max" "$l2_max" | sort -gr | head -1)

                    # collision rate
                    collision=$(grep "Collision_rate:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}')

                    echo "[$(date)] $arm epoch=$cur_epoch CHECK @ ep=$check_ep: L0_max=$l0_max L1_max=$l1_max L2_max=$l2_max collision=$collision"

                    # Gate 1: hyp_norm_max > 0.98 (B1 失效)
                    if awk -v v="$overall_max" 'BEGIN {exit !(v > 0.98)}'; then
                        echo "[$(date)] ❌ $arm ABORT @ epoch $cur_epoch: hyp_norm_max=$overall_max > 0.98 (boundary saturation, B1 失效模式)"
                        echo "epoch=$cur_epoch hyp_norm_max=$overall_max collision=$collision reason=boundary_saturation" > $ABORT_FLAG_DIR/$arm
                        kill -TERM $pid 2>/dev/null
                        sleep 3
                        kill -KILL $pid 2>/dev/null
                        rm -f $pid_file
                        continue
                    fi

                    # Gate 2: dyn_range < 1.5 (A3 失效) - 暂时无法从 train_hrqvae.py log 直接读, 跳过
                    # 用户说 "epoch 20 + 50 检查", 当前只能看 hypnorm. dyn 检查需要修改 train script (上游)
                    # R11.4: 不动上游, 只用 hyp gate. dyn 检查等训练完再做.
                fi
            fi
        done
    done
done
echo "[$(date)] 监控 loop 结束 (120 轮 × 60s = 2h)"
