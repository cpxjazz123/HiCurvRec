#!/bin/bash
# Issue #21 Gate 1 — launcher header 约束 + precedent_override 强制模板
# 2026-07-29
#
# 解决问题: Issue #20 越闸显示 Issue #17/#18/#19 三层防护全部失守
#          (工具修完了, 但 R11.4 自主决策用 proxy + precedent 跨过 §6.7.4 stop-loss (i))
#          本模板强制 launcher 在头部声明:
#            # GATE_DECLARATION:
#            #   section_6_7_4_stop_loss_i: <bound|not_bound>
#            #   stage_2_threshold:        <numeric or NA>
#            #   stage_3_threshold:        <numeric or NA>
#            #   auto_proceed_after_stage_2: <true|false>
#            #   precedent_override:       <allowed|forbidden>
#          并在 Stage 2 → Stage 3 之间强制求值, 不通过则非零退出.
#
# 默认 precedent_override=forbidden. allowed 必须由 owner 显式批准
# (verdicts/<owner_approval>.md 文件 + 该文件含 launcher path 字符串).
#
# 通过条件 (Issue #21 Gate 1 三条同时):
#   (a) task275_a2_extend.sh 回放 + L0=89.1% → 模板在 Stage 3 之前非零退出
#   (b) task237_arm_b_full_chain.sh 回放 (Issue #19 已处理) → 模板在 Stage 3 之前退出
#   (c) 人为构造通过案例 (header not_bound + auto_proceed=true) → 模板放行
#
# 用法 (作为链式 launcher 的一部分):
#   source scripts/issue21_gate_header.sh
#   parse_gate_declaration <launcher_path> <stage1_log>
#   if [ $? -eq 0 ]; then
#       run_stage_3 ...
#   fi
#
# 独立回放 (Gate 1 验证):
#   bash scripts/issue21_gate_header.sh <launcher_path> <stage1_log>
#   exit code: 0 = PASS (放行 Stage 3), 1 = FAIL (退出, 不放行), 2 = ERROR (缺文件/字段)

set -u

# 闸门定义 (可被环境变量 override)
GATE_L0_THRESHOLD="${GATE_L0_THRESHOLD:-90}"                  # §6.7.4 stop-loss (i) L0 ≥ 90%
GATE_EPOCH_MIN="${GATE_EPOCH_MIN:-30}"                        # L0 测量必须在 epoch ≥ 30 (warmup)
GATE_PRECEDENT_DEFAULT="${GATE_PRECEDENT_DEFAULT:-forbidden}"  # 默认 precedent_override=forbidden
GATE_AUTO_PROCEED_DEFAULT="${GATE_AUTO_PROCEED_DEFAULT:-false}" # 默认 auto_proceed_after_stage_2=false (要求显式求值)

# 入参: $1 = launcher 路径 (含 # GATE_DECLARATION: 块)
#       $2 = Stage 1 log 路径 (hrqvae.log, 含 L0 utilization)
# 行为: parse declaration → check §6.7.4 stop-loss (i) → exit 0/1/2
# 返回: 0 = PASS (放行 Stage 3), 1 = FAIL (退出, 不放行), 2 = ERROR (缺文件/字段)
parse_gate_declaration() {
    local launcher_path="$1"
    local stage1_log="$2"

    if [ ! -f "$launcher_path" ]; then
        echo "[Issue #21] ❌ ERROR: launcher 不存在: $launcher_path"
        return 2
    fi

    # 解析 # GATE_DECLARATION: 块 (前 30 行内)
    local decl=$(head -30 "$launcher_path" | grep -A 6 "^# GATE_DECLARATION:" | tail -6)

    if [ -z "$decl" ]; then
        echo "[Issue #21] ❌ FAIL: launcher 缺 # GATE_DECLARATION: 块"
        echo "  launcher: $launcher_path"
        echo "  模板要求 launcher 在前 30 行声明 GATE_DECLARATION 五字段"
        return 1
    fi

    local s6_7_4=$(echo "$decl" | grep "section_6_7_4_stop_loss_i:" | awk '{print $NF}')
    local stage2_th=$(echo "$decl" | grep "stage_2_threshold:" | awk '{print $NF}')
    local stage3_th=$(echo "$decl" | grep "stage_3_threshold:" | awk '{print $NF}')
    local auto_proceed=$(echo "$decl" | grep "auto_proceed_after_stage_2:" | awk '{print $NF}')
    local precedent=$(echo "$decl" | grep "precedent_override:" | awk '{print $NF}')

    # 默认值 (R11.2 保守兜底: 显式声明优先, 不显式则用最严格默认值)
    s6_7_4=${s6_7_4:-bound}
    auto_proceed=${auto_proceed:-$GATE_AUTO_PROCEED_DEFAULT}
    precedent=${precedent:-$GATE_PRECEDENT_DEFAULT}

    echo "[Issue #21] 解析声明:"
    echo "  section_6_7_4_stop_loss_i = $s6_7_4"
    echo "  stage_2_threshold          = ${stage2_th:-NA}"
    echo "  stage_3_threshold          = ${stage3_th:-NA}"
    echo "  auto_proceed_after_stage_2 = $auto_proceed"
    echo "  precedent_override         = $precedent"
    echo "  ---"

    # 1. precedent_override 校验 (R11.4 自主决策权 vs Issue #21 治理边界)
    if [ "$precedent" = "allowed" ]; then
        # 必须有 owner 显式批准 verdict (verdicts/<owner_approval>.md 引用本 launcher path)
        local approval=$(grep -rl "$(basename "$launcher_path")" verdicts/ 2>/dev/null | head -1)
        if [ -z "$approval" ]; then
            echo "[Issue #21] ❌ FAIL: precedent_override=allowed 但无 owner approval verdict"
            echo "  必须存在 verdicts/<owner_approval>.md 引用本 launcher path"
            return 1
        fi
        echo "  precedent_override=allowed (owner approval: $approval)"
    fi

    # 2. §6.7.4 stop-loss (i) 校验 (only if bound AND auto_proceed=false)
    if [ "$s6_7_4" = "bound" ] && [ "$auto_proceed" = "false" ]; then
        # 必须有 Stage 1 log
        if [ ! -f "$stage1_log" ]; then
            echo "[Issue #21] ❌ FAIL: §6.7.4 stop-loss (i) bound + auto_proceed=false 但 Stage 1 log 不存在"
            echo "  stage1_log: $stage1_log"
            echo "  必须先有 Stage 1 训练 + L0 utilization 证据, 才能进 Stage 2/3"
            return 1
        fi

        # 提取 epoch ≥ $GATE_EPOCH_MIN 范围内最后一个 L0 utilization
        # 注意: regex 必须用 \[step2 monitor ep[0-9]+\] 避免误匹配 "step" 里的 "ep"
        local l0_line=$(awk -v min_ep="$GATE_EPOCH_MIN" '
            /step2 monitor ep/ {
                match($0, /\[step2 monitor ep([0-9]+)\]/, arr)
                ep = arr[1] + 0
                if (ep >= min_ep) {
                    match($0, /usage=([0-9.]+)%/, m)
                    if (m[1] != "") {
                        last_l0 = m[1]
                        last_ep = ep
                    }
                }
            }
            END {
                if (last_l0 != "") print last_ep, last_l0
            }
        ' "$stage1_log")

        if [ -z "$l0_line" ]; then
            echo "[Issue #21] ❌ FAIL: Stage 1 log 缺 epoch ≥ ${GATE_EPOCH_MIN} 的 L0 utilization"
            echo "  Issue #17 修复保证 L0 打印, 如缺则说明 Stage 1 log 异常或 trainer nested-scope 退化"
            echo "  必须先解决 L0 打印问题, 不能用 collision_rate 作为 L0 proxy 跨过"
            return 1
        fi

        local l0_ep=$(echo "$l0_line" | awk '{print $1}')
        local l0_pct=$(echo "$l0_line" | awk '{print $2}')

        # L0 ≥ 90% 判据
        local l0_int=$(python3 -c "print(int(float('$l0_pct')))")
        echo "  Stage 1 L0 (epoch ${l0_ep}) = ${l0_int}%   (阈值 ≥ ${GATE_L0_THRESHOLD}%)"

        if [ "$l0_int" -lt "$GATE_L0_THRESHOLD" ]; then
            echo "[Issue #21] ❌ FAIL: §6.7.4 stop-loss (i) 触发"
            echo "  L0 = ${l0_int}% < ${GATE_L0_THRESHOLD}%, 禁止自动进入 Stage 3"
            echo "  不接受任何 proxy / precedent / Task #178 类比 (precedent_override=$precedent)"
            return 1
        fi

        echo "  ✅ §6.7.4 stop-loss (i) 未触发 (L0 ${l0_int}% >= ${GATE_L0_THRESHOLD}%)"
        echo "[Issue #21] ✅ PASS — §6.7.4 stop-loss (i) 通过, 放行 Stage 3"
        return 0
    fi

    # auto_proceed=true 或 not_bound → 放行
    echo "  auto_proceed=$auto_proceed, bound=$s6_7_4 → header 声明放行 (无需 L0 求值)"
    echo "[Issue #21] ✅ PASS — header 声明放行"
    return 0
}

# CLI mode: 独立运行回放测试
if [ "${#@}" -gt 0 ]; then
    parse_gate_declaration "$@"
    exit $?
fi
