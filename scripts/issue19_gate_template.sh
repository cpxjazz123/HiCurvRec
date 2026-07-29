#!/bin/bash
# Issue #19 Gate 1 — 链式 launcher 通用闸门求值模板
# 2026-07-29
#
# 解决问题: Stage 2 → Stage 3 链式 launcher 没有"在 Stage 3 启动前求值 Stage 2 判据"的退出点.
#          本模板在 Stage 2 推断**之后**、Stage 3 训练**之前**插入一次闸门求值, 不通过则非零退出.
#          通过则正常放行 (Gate 1 (c) 用例).
#
# 通过条件 (Issue #19 Gate 1 通过条件, 三条同时):
#   (a) task237 历史产物回放 → 模板早退 (exit code ≠ 0)
#   (b) task260 max_iters=30 产物回放 → 同样早退
#   (c) 通过案例回放 → 模板正常放行
#
# 用法 (作为链式 launcher 的一部分):
#   source scripts/issue19_gate_template.sh
#   run_stage_2 ...
#   evaluate_stage_2_gate     # ← 插入此调用
#   if [ $? -eq 0 ]; then
#       run_stage_3 ...
#   fi
#
# 或独立运行回放测试 (Gate 1 验证):
#   bash scripts/issue19_gate_template.sh <stage2_diag.json> <a_col> <c_col>
#   exit code: 0 = PASS (放行 Stage 3), 1+ = FAIL (退出)

set -u

# 闸门定义 (可被环境变量 override)
# 默认基于 Issue #10 Gate 1: Arm A = 0.99 (vanilla), Arm C = 0.05 (full Sinkhorn)
GATE_A_COL="${GATE_A_COL:-0.99}"      # Arm A 期望 collision_pre_resolve (Issue #10 §vanilla baseline)
GATE_C_COL="${GATE_C_COL:-0.05}"      # Arm C 期望 collision_pre_resolve (Issue #10 §full Sinkhorn)
GATE_DIFF_PP="${GATE_DIFF_PP:-15}"    # 判据: |B-A| ≥ 15pp AND |B-C| ≥ 15pp (Issue #10 §gate 阈值)
GATE_NO_STAGE2_BLOCK="${GATE_NO_STAGE2_BLOCK:-0}"  # 显式声明无 Stage 2 级闸门 (e.g., ROUND 4 R@10 之类)

# 求值函数
# 入参: $1 = path to stage2_diagnostic.json
# 行为: echo 出判据 + 写 verdict 草稿 + 设置 EXIT_CODE
# 返回: 0 if PASS, 1 if FAIL, 2 if 错误 (e.g., 文件缺失)
evaluate_stage_2_gate() {
    local stage2_json="$1"

    if [ "$GATE_NO_STAGE2_BLOCK" = "1" ]; then
        echo "[Issue #19] GATE_NO_STAGE2_BLOCK=1, 显式声明无 Stage 2 级闸门, 放行"
        exit 0
    fi

    if [ ! -f "$stage2_json" ]; then
        echo "[Issue #19] ❌ ERROR: Stage 2 诊断 JSON 不存在: $stage2_json"
        exit 2
    fi

    # 读 collision_rate_pre_resolve (考虑 task237 vs task260 不同 key)
    local B=$(python3 -c "
import json, sys
d = json.load(open('$stage2_json'))
for k in ['collision_rate_pre_resolve', 'collision_pre_resolve', 'B_collision_pre_resolve']:
    if k in d:
        print(d[k])
        sys.exit(0)
print('MISSING')
sys.exit(1)
")
    if [ "$B" = "MISSING" ]; then
        echo "[Issue #19] ❌ ERROR: 诊断 JSON 缺 collision_pre_resolve 字段"
        exit 2
    fi

    # 算 |B-A| 和 |B-C|
    local diff_a=$(python3 -c "print(abs(float('$B') - float('$GATE_A_COL')))")
    local diff_c=$(python3 -c "print(abs(float('$B') - float('$GATE_C_COL')))")
    local diff_a_pp=$(python3 -c "print(int(float('$diff_a') * 100))")
    local diff_c_pp=$(python3 -c "print(int(float('$diff_c') * 100))")

    # 检查 L0/L1/L2 利用率 (100% 为 OK)
    local l0=$(python3 -c "
import json
d = json.load(open('$stage2_json'))
if 'per_layer_utilization' in d:
    p = d['per_layer_utilization']
    if 'L0' in p:
        print(int(p['L0'].get('utilization', p['L0'].get('fraction', 0)) * 100))
    elif 'layer_0' in p:
        print(int(p['layer_0'].get('fraction', 0) * 100))
    else:
        print('NA')
else:
    print('NA')
")

    # 判据
    local pass_a="N"; local pass_c="N"
    [ "$diff_a_pp" -ge "$GATE_DIFF_PP" ] && pass_a="Y"
    [ "$diff_c_pp" -ge "$GATE_DIFF_PP" ] && pass_c="Y"

    echo "============================================="
    echo "[Issue #19 Gate 1] Stage 2 闸门求值"
    echo "  Source: $stage2_json"
    echo "  B (Arm B collision_pre) = $B"
    echo "  |B-A| = $diff_a_pp pp    (Arm A = $GATE_A_COL, 阈值 ≥ $GATE_DIFF_PP pp)"
    echo "  |B-C| = $diff_c_pp pp    (Arm C = $GATE_C_COL, 阈值 ≥ $GATE_DIFF_PP pp)"
    echo "  L0 utilization = $l0 %"
    echo "  -- 判据 --"
    echo "  |B-A| >= $GATE_DIFF_PP pp: $pass_a"
    echo "  |B-C| >= $GATE_DIFF_PP pp: $pass_c"
    echo "============================================="

    # Issue #10 Gate 1 要两条都满足 → 否则不构成 3-arm 曲线 → 退出 1
    if [ "$pass_a" = "Y" ] && [ "$pass_c" = "Y" ]; then
        echo "[Issue #19 Gate 1] ✅ PASS — 两条都满足, 放行 Stage 3"
        exit 0
    else
        echo "[Issue #19 Gate 1] ❌ FAIL — 至少一条不满足, 不放行 Stage 3"
        exit 1
    fi
}

# 如果作为独立脚本调用 (命令行带参数), 跑一次 evaluate
if [ "${#@}" -gt 0 ]; then
    evaluate_stage_2_gate "$@"
fi
