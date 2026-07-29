#!/bin/bash
# Task #290 — Issue #21 Gate 1 (c) 人为构造通过案例 (sanity check)
# 2026-07-29
#
# GATE_DECLARATION:
#   section_6_7_4_stop_loss_i: not_bound
#   stage_2_threshold:        NA
#   stage_3_threshold:        NA
#   auto_proceed_after_stage_2: true
#   precedent_override:       forbidden
#
# 目的: 验证 Issue #21 模板在 header 声明 not_bound + auto_proceed=true 时正确放行
#       (即, 当 owner 显式声明 "本 launcher 不受 §6.7.4 stop-loss (i) 约束, 自动放行"
#        且 precedent_override=forbidden 时, 模板不应阻拦). 这是 Gate 1 (c) 通过案例.
#
# 注意: precedent_override=forbidden 在 not_bound 时仍然要求 owner 没有用 proxy / precedent
#       跨过. 这里 header 显式声明 not_bound 是合法的 owner 声明, 不算 precedent 跨过.
#       (precedent_override=allowed 才是允许用 proxy / precedent 类比跨过; 当前 default 是
#        forbidden, 必须 owner 显式 verdict 批准)

set -u

# 这个 launcher 本身不启动任何 Stage 2/3, 仅用于回放模板测试
echo "[$(date)] Issue #21 Gate 1 (c) pass case launcher"
echo "[$(date)] header declares not_bound + auto_proceed=true + precedent_override=forbidden"
echo "[$(date)] 期望: parse_gate_declaration 应返回 exit 0 (放行)"

# 模拟 Issue #21 模板调用
source /home/wlia0047/ar57/wenyu/GeneRec/scripts/issue21_gate_header.sh
parse_gate_declaration "$0" "/dev/null"
exit_code=$?
echo "[$(date)] parse_gate_declaration exit code = $exit_code"

if [ "$exit_code" = "0" ]; then
    echo "[$(date)] ✅ Gate 1 (c) PASS — 模板在 not_bound + auto_proceed=true 时正确放行"
    exit 0
else
    echo "[$(date)] ❌ Gate 1 (c) FAIL — 模板应放行但 exit=$exit_code"
    exit 1
fi
