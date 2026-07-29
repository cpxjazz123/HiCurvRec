#!/bin/bash
# Issue #21 Gate 2 — 越闸计数暴露 (cron tick 启动日志 + 模板调用双 hook)
# 2026-07-29
#
# 解决问题: Issue #17/#18/#19 修了工具层, Issue #20 显示治理层越闸仍能发生
#          (R11.4 proxy + precedent). 但越闸次数不暴露在常规日志里, owner 不易察觉.
#          本脚本强制在 cron tick 启动 + 模板被调用时, 把越闸次数 + 最近 verdict 路径
#          打到日志头部. 格式固定, 可被 grep 解析.
#
# 通过条件 (Issue #21 Gate 2):
#   日志格式固定 (prefix [Issue #21 Skip Counter]), 可被 grep 解析
#
# 用法:
#   1. cron tick 启动: source scripts/issue21_skip_counter.sh; emit_skip_counter
#   2. 模板调用: 顶部自动 emit (无需显式调用)
#   3. 独立跑: bash scripts/issue21_skip_counter.sh

set -u

# 闸门真源: verdicts/task_gate_skip_history_v6.md (Gate 0 闭环产物)
# 计数 = Gate 0 表格中的 6 次越闸 (固定, 直到新的越闸发生并写入 v7+)
SKIP_HISTORY_FILE="${SKIP_HISTORY_FILE:-/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task_gate_skip_history_v6.md}"
GATE_DECLARATION_LOG="${GATE_DECLARATION_LOG:-/home/wlia0047/ar57/wenyu/GeneRec/logs/issue21_skip_counter.log}"

emit_skip_counter() {
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    local tick_id=$(date '+%Y%m%d_%H%M%S')

    # 解析 Gate 0 历史越闸记录
    local total_skips=0
    local last_verdict="NONE"
    local last_date="NONE"

    if [ -f "$SKIP_HISTORY_FILE" ]; then
        # 计数 = "次序 N" 的行数
        total_skips=$(grep -cE '^\| \*\*[0-9]+\*\*' "$SKIP_HISTORY_FILE" 2>/dev/null || echo 0)
        # 最近 verdict: 汇总表最后一行 "出处 verdict" 列 (次序 N 最大的行)
        last_verdict=$(grep -E '^\| \*\*[0-9]+\*\*' "$SKIP_HISTORY_FILE" 2>/dev/null \
            | tail -1 \
            | grep -oE 'verdicts/task[0-9_a-z]+\.md' \
            | head -1)
        if [ -z "$last_verdict" ]; then
            last_verdict="NONE"
        fi
    fi

    # Issue #21 Gate 0/1 闭环 (本任务自身进展)
    local gate0_status="OPEN"
    local gate1_status="OPEN"
    if [ -f "/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task_gate_skip_history_v6.md" ]; then
        gate0_status="CLOSED"
    fi
    if [ -f "/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task290_issue21_gate1_result.md" ]; then
        gate1_status="CLOSED"
    fi

    # 固定格式输出 (Issue #21 Gate 2 通过条件: 日志格式固定, 可被 grep 解析)
    local line1="[Issue #21 Skip Counter] tick_id=${tick_id} timestamp=${ts} total_skips=${total_skips} last_verdict=${last_verdict} gate0=${gate0_status} gate1=${gate1_status}"

    echo "$line1"

    # 写日志 (供 cron tick + 审计 grep)
    if [ -n "${GATE_DECLARATION_LOG:-}" ]; then
        mkdir -p "$(dirname "$GATE_DECLARATION_LOG")" 2>/dev/null || true
        echo "$line1" >> "$GATE_DECLARATION_LOG" 2>/dev/null || true
    fi
}

# CLI mode: 独立运行
if [ "${#@}" -eq 0 ]; then
    emit_skip_counter
fi
