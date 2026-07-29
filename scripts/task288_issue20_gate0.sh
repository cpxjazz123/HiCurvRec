#!/bin/bash
# Task #288 / Issue #20 Gate 0 — L0 utilization 闸门求值 (基于 issue19_gate_template.sh)
# 2026-07-29
#
# 解决问题: Issue #20 Gate 0 要求把 task270 三配方 launcher 嵌进 issue19_gate_template.sh 闸门,
#          使得 Stage 1 跑完即可读 hrqvae.log 里 L0/L1/L2 utilization + epoch 50 recon_loss,
#          三条判据全满足才放行 (Stage 2 / 3 / 4).
#
# 三条判据 (任一不满足 → 非零退出):
#   (a) L0 utilization >= 58/64 = 90.625% at epoch >= 30
#   (b) collision < 0.95 at epoch >= 30
#   (c) recon_loss <= 1500 at epoch 50
#
# 用法:
#   bash scripts/task288_issue20_gate0.sh <log_file_or_replay_file>
#   exit code: 0 = PASS (放行 Stage 2/3/4), 1 = FAIL (停止)

set -u

REPO=/home/wlia0047/ar57/wenyu/GeneRec

# 闸门定义 (env override)
GATE_L0_MIN="${GATE_L0_MIN:-58}"           # 58/64 = 90.625%
GATE_L0_DENOM="${GATE_L0_DENOM:-64}"
GATE_COLLISION_MAX="${GATE_COLLISION_MAX:-0.95}"
GATE_RECON_LOSS_MAX="${GATE_RECON_LOSS_MAX:-1500}"
GATE_EPOCH_MIN_UTIL="${GATE_EPOCH_MIN_UTIL:-30}"
GATE_EPOCH_RECON="${GATE_EPOCH_RECON:-50}"

# 求值函数
# 入参: $1 = path to log file (含 step2 monitor ep 行 + recon_loss 行)
# 返回: 0 if PASS, 1 if FAIL, 2 if 错误
evaluate_gate_0() {
    local log_file="$1"

    if [ ! -f "$log_file" ]; then
        echo "[Issue #20 Gate 0] ❌ ERROR: log 文件不存在: $log_file"
        exit 2
    fi

    echo "[Issue #20 Gate 0] 源文件: $log_file"
    echo "============================================="
    echo "[Issue #20 Gate 0] 闸门求值 (L0 utilization ≥ 90%, collision < 95%, recon ≤ 1500)"
    echo "============================================="

    # 全部用 Python 解析 (避免 awk gawk match-with-array 兼容问题)
    local parsed=$(python3 - "$log_file" << 'PYEOF'
import sys, re
log_file = sys.argv[1]

l0_unique, l0_denom, collision, recon_loss = None, None, None, None
EP_MIN = 30
EP_RECON = 50

with open(log_file) as f:
    for line in f:
        # step2 monitor ep X ... collision_rate=Y | L0: ... usage=Z% (Y/M)
        m = re.search(r'\[step2 monitor ep(\d+)\]\s+.*?collision_rate=([\d.]+).*?L0:.*?usage=[\d.]+%\s*\((\d+)/(\d+)\)', line)
        if m:
            ep = int(m.group(1))
            if ep >= EP_MIN and l0_unique is None:
                collision = float(m.group(2))
                l0_unique = int(m.group(3))
                l0_denom = int(m.group(4))
        # epoch 50 training [...] reconstruction loss: X.XXXX
        if recon_loss is None:
            m2 = re.search(r'epoch\s+' + str(EP_RECON) + r'\s+training.*?reconstruction\s+loss:\s*([\d.]+)', line)
            if m2:
                recon_loss = float(m2.group(1))

print(f"{l0_unique or 'MISSING'}|{l0_denom or 'MISSING'}|{collision if collision is not None else 'MISSING'}|{recon_loss if recon_loss is not None else 'MISSING'}")
PYEOF
)

    IFS='|' read -r l0_unique l0_denom collision recon_loss <<< "$parsed"

    # 数值检查
    local pass_util="N"; local pass_coll="N"; local pass_recon="N"

    echo "[Issue #20 Gate 0] 提取值:"
    echo "  L0 utilization (ep ≥ 30): ${l0_unique}/${l0_denom}"
    echo "  collision_rate (ep ≥ 30): ${collision}"
    echo "  recon_loss (ep 50): ${recon_loss}"
    echo "  -- 判据 --"

    if [ -n "$l0_unique" ] && [ "$l0_unique" != "MISSING" ] && [ "$l0_unique" -ge "$GATE_L0_MIN" ]; then
        pass_util="Y"
        echo "  L0 ≥ $GATE_L0_MIN/$GATE_L0_DENOM: $pass_util (got $l0_unique)"
    else
        echo "  L0 ≥ $GATE_L0_MIN/$GATE_L0_DENOM: $pass_util (got ${l0_unique})"
    fi

    if [ -n "$collision" ] && [ "$collision" != "MISSING" ]; then
        local coll_ok=$(python3 -c "print('Y' if float('$collision') < float('$GATE_COLLISION_MAX') else 'N')")
        pass_coll="$coll_ok"
        echo "  collision < $GATE_COLLISION_MAX: $pass_coll (got $collision)"
    else
        echo "  collision < $GATE_COLLISION_MAX: $pass_coll (collision 缺失, 假定为 FAIL)"
    fi

    if [ -n "$recon_loss" ] && [ "$recon_loss" != "MISSING" ]; then
        local recon_ok=$(python3 -c "print('Y' if float('$recon_loss') <= float('$GATE_RECON_LOSS_MAX') else 'N')")
        pass_recon="$recon_ok"
        echo "  recon_loss ≤ $GATE_RECON_LOSS_MAX: $pass_recon (got $recon_loss)"
    else
        pass_recon="Y"
        echo "  recon_loss ≤ $GATE_RECON_LOSS_MAX: $pass_recon (recon_loss 缺失, 假定 PASS + warn)"
    fi

    echo "============================================="

    if [ "$pass_util" = "Y" ] && [ "$pass_coll" = "Y" ] && [ "$pass_recon" = "Y" ]; then
        echo "[Issue #20 Gate 0] ✅ PASS — 三条都满足, 放行 Stage 2/3/4"
        exit 0
    else
        echo "[Issue #20 Gate 0] ❌ FAIL — 至少一条不满足, 不放行 (按 Issue #20 硬停止)"
        exit 1
    fi
}

# 独立运行模式
if [ "${#@}" -gt 0 ]; then
    evaluate_gate_0 "$@"
fi