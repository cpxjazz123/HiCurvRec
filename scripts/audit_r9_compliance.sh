#!/usr/bin/env bash
# scripts/audit_r9_compliance.sh
#
# R9 合规审计 (CLAUDE.md R9-Enforce 层 3)
# 每个 loop tick 第一步必跑. 返回 0 = pass, 1 = fail.
#
# 检查项:
# 1. descriptions/ 任务编号连续无空洞 (强制)
# 2. verdicts/ 任务编号连续 (warning only, 允许历史空洞)
# 3. 残余 task<old_num> 引用检查 (除 renumber 脚本自身 + 当前 loop.md §16 外, 不应有 100-200 范围的旧编号)
#
# 用法: bash scripts/audit_r9_compliance.sh
# 输出: R9 compliance status + 详细诊断

# 不使用 set -e: grep 在没有匹配时返回 1, 这不是错误.
# 改用显式逻辑控制 PASS / FAIL 计数.

REPO_ROOT="/home/wlia0047/ar57/wenyu/GeneRec"
DESCRIPTIONS_DIR="$REPO_ROOT/descriptions"
VERDICTS_DIR="$REPO_ROOT/verdicts"
SCRIPTS_DIR="$REPO_ROOT/scripts"
LOGS_DIR="$REPO_ROOT/logs"
LOOP_MD="$REPO_ROOT/loop.md"

PASS=0
FAIL=0

echo "==============================================="
echo "R9 合规审计 (CLAUDE.md R9-Enforce)"
echo "==============================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ----------------------------------------------------------------------
# 检查 1: descriptions/ 任务编号连续无空洞 (强制)
# ----------------------------------------------------------------------
echo "[1] descriptions/ 任务编号连续性检查"
if [ ! -d "$DESCRIPTIONS_DIR" ]; then
    echo "  ❌ descriptions/ 不存在"
    FAIL=$((FAIL + 1))
else
    # 使用 || true 防止 grep 无匹配时返回 1 提前终止
    ACTUAL_IDS=$(ls "$DESCRIPTIONS_DIR" 2>/dev/null | grep -oE 'task[0-9]+' | grep -oE '[0-9]+' | sort -n | uniq | grep -v '^$' || true)
    MAX_ID=$(echo "$ACTUAL_IDS" | grep -v '^$' | tail -1 || true)

    if [ -z "$ACTUAL_IDS" ]; then
        echo "  ⚠️ descriptions/ 为空 (没有任何任务)"
        FAIL=$((FAIL + 1))
    else
        EXPECTED_FILE=$(mktemp)
        ACTUAL_FILE=$(mktemp)
        awk -v n="$MAX_ID" 'BEGIN{for(i=1;i<=n;i++) print i}' > "$EXPECTED_FILE"
        echo "$ACTUAL_IDS" | sort -n > "$ACTUAL_FILE"
        # 使用 diff 而非 comm: diff 对输入顺序检查更宽容, 且给出相同结果
        # diff 期望 lexicographic 排序, 这里两个文件都是数字行, sort -n 给出 lexicographic 顺序
        GAPS=$(diff "$EXPECTED_FILE" "$ACTUAL_FILE" | grep '^<' | awk '{print $2}' | grep -v '^$' || true)
        rm -f "$EXPECTED_FILE" "$ACTUAL_FILE"

        if [ -z "$GAPS" ]; then
            N_ACTUAL=$(echo "$ACTUAL_IDS" | wc -l)
            echo "  ✅ descriptions/ 连续无空洞 (max=$MAX_ID, 共 $N_ACTUAL 个任务)"
            PASS=$((PASS + 1))
        else
            echo "  ❌ descriptions/ 存在空洞:"
            echo "      空洞编号: $(echo $GAPS | tr '\n' ' ')"
            echo "      必须立即用 scripts/renumber_tasks_*.py 填补"
            echo "      下次新任务编号 = $((MAX_ID + 1)) (但严禁写入, 必须先填空)"
            FAIL=$((FAIL + 1))
        fi
    fi
fi
echo ""

# ----------------------------------------------------------------------
# 检查 2: verdicts/ 任务编号连续性 (warning)
# ----------------------------------------------------------------------
echo "[2] verdicts/ 任务编号连续性检查 (warning)"
if [ ! -d "$VERDICTS_DIR" ]; then
    echo "  ⚠️ verdicts/ 不存在"
else
    ACTUAL_IDS=$(ls "$VERDICTS_DIR" 2>/dev/null | grep -oE 'task[0-9]+' | grep -oE '[0-9]+' | sort -n | uniq | grep -v '^$' || true)
    N_VERDICTS=$(echo "$ACTUAL_IDS" | grep -v '^$' | wc -l | tr -d ' ')
    N_DESCRIPTIONS=$(ls "$DESCRIPTIONS_DIR" 2>/dev/null | grep -oE 'task[0-9]+' | wc -l | tr -d ' ')

    if [ -z "$ACTUAL_IDS" ]; then
        echo "  ℹ️ verdicts/ 为空"
    else
        MAX_ID=$(echo "$ACTUAL_IDS" | grep -v '^$' | tail -1 || true)
        EXPECTED_FILE=$(mktemp)
        ACTUAL_FILE=$(mktemp)
        awk -v n="$MAX_ID" 'BEGIN{for(i=1;i<=n;i++) print i}' > "$EXPECTED_FILE"
        echo "$ACTUAL_IDS" | sort -n > "$ACTUAL_FILE"
        GAPS=$(diff "$EXPECTED_FILE" "$ACTUAL_FILE" | grep '^<' | awk '{print $2}' | grep -v '^$' | head -10 || true)
        rm -f "$EXPECTED_FILE" "$ACTUAL_FILE"

        if [ -z "$GAPS" ]; then
            echo "  ✅ verdicts/ 连续无空洞 (max=$MAX_ID, 共 $N_VERDICTS 个 verdict)"
        else
            echo "  ⚠️ verdicts/ 有历史空洞 (允许): $(echo $GAPS | tr '\n' ' ' | head -c 100)"
            echo "      (历史遗留, 不在 R9-Enforce 强制范围; 但如果新任务导致空洞, 必须填补)"
        fi
    fi
    echo "  ℹ️  verdict 数量 = $N_VERDICTS, description 数量 = $N_DESCRIPTIONS (差值 = 分析任务无 description)"
fi
echo ""

# ----------------------------------------------------------------------
# 检查 3: 残余 task<old_num> 引用 (除已知白名单外)
# ----------------------------------------------------------------------
echo "[3] 残余 task<old_num> 引用检查"
# 严格语义: 检测的应是"被 renumber 过的编号 (104-130) 但当前已不存在对应新编号文件"的引用.
# 因为 task55 / task281 是历史上从未在 renumber 范围内的合法旧编号 (35 之后 Blocked by user policy),
# 不应算 FAIL (它们是历史合法引用).
#
# 真正的 renumber 残留范围: task104-task130 (在 renumber_tasks_104_to_23.py 中已重编号到 23-35 / 32-35)

# 构造"已知合法新编号"白名单 (descriptions/ 当前实际存在)
WHITELIST=$(ls "$DESCRIPTIONS_DIR" 2>/dev/null | grep -oE '^task[0-9]+' | sort -u)
WHITELIST_VERDICTS=$(ls "$VERDICTS_DIR" 2>/dev/null | grep -oE '^task[0-9]+' | sort -u)

# 真正被 renumber 的范围: 104-117 → 23-35, 127-130 → 32-35 (Task #80 + #133)
# task118-126 不在 MAPPING 内, 是合法当前编号 (e.g. task119, task120, task123, task125, task126)
# 注意: task105/107/109/110/111/112 是 post-renumber 合法有效任务 (paper-defense artifacts), 引用合法.
# 严格语义: 残余 = 引用编号 ∈ RENUMBERED_NUMBERS 但 not in WHITELIST (descriptions/ 当前实际存在)
RENUMBERED_NUMBERS="task104|task105|task107|task108|task109|task110|task111|task112|task113|task114|task115|task116|task117|task127|task128|task129|task130"
RESIDUAL_RENUMBER=$(grep -lrE "$RENUMBERED_NUMBERS" \
    "$DESCRIPTIONS_DIR" "$VERDICTS_DIR" "$SCRIPTS_DIR" \
    2>/dev/null | \
    grep -vE "/(renumber_tasks_|fix_old_task_refs_in_logs|audit_r9_compliance|task134_r9_enforce_audit)" || true)
# 单独检查 logs/ 顶层 task*.log 文件 (非 .hydra/ 元数据)
RESIDUAL_LOGS=$(grep -lE "task(1(0[4-9]|1[0-9]|2[0-9]|30))" "$LOGS_DIR"/task*.log 2>/dev/null | grep -v "renumber_tasks_" || true)
RESIDUAL_RENUMBER=$(printf "%s\n%s\n" "$RESIDUAL_RENUMBER" "$RESIDUAL_LOGS" | grep -v '^$' | sort -u || true)

# 排除每个文件中自指 (文件名前缀匹配) + 当前有效引用 (在 WHITELIST 中的不是残余)
REAL_RESIDUAL=""
if [ -n "$RESIDUAL_RENUMBER" ]; then
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        # 提取每个文件中所有 task<NUM> 引用, 检查是否在 RENUMBERED_NUMBERS 中
        refs=$(grep -oE "$RENUMBERED_NUMBERS" "$f" 2>/dev/null | sort -u || true)
        for ref in $refs; do
            # 跳过自指 (verdict 文件 task125_xxx.md 里出现 task125 是合法的)
            fname_self=$(basename "$f" | grep -oE '^task[0-9]+')
            if [ "$fname_self" = "$ref" ]; then
                continue
            fi
            # 跳过当前有效引用 (task105/110/111 等是 post-renumber 合法任务, descriptions/ 中存在)
            if echo "$WHITELIST" | grep -q "^${ref}$"; then
                continue
            fi
            # 否则这是潜在残留, 加入 REAL_RESIDUAL
            REAL_RESIDUAL="$REAL_RESIDUAL $f:$ref"
            break
        done
    done <<< "$RESIDUAL_RENUMBER"
fi
REAL_RESIDUAL=$(echo "$REAL_RESIDUAL" | xargs -n1 2>/dev/null | sort -u | head -10 || true)

if [ -z "$REAL_RESIDUAL" ]; then
    echo "  ✅ 无 task104-130 范围 renumber 残留 (排除自指)"
    PASS=$((PASS + 1))
else
    echo "  ❌ 发现 renumber 残留 (排除自指 + 排除 renumber 脚本自身):"
    echo "$REAL_RESIDUAL" | head -5 | while read line; do
        [ -z "$line" ] && continue
        f=$(echo "$line" | cut -d: -f1)
        ref=$(echo "$line" | cut -d: -f2)
        echo "      - $f: 引用 $ref"
    done
    FAIL=$((FAIL + 1))
fi
echo ""

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
echo "==============================================="
echo "R9 合规审计结果"
echo "==============================================="
if [ $FAIL -gt 0 ]; then
    echo "❌ FAIL: $FAIL 项不通过, $PASS 项通过"
    echo "   必须立即修复 FAIL 项, 然后再继续任何任务"
    exit 1
else
    echo "✅ PASS: $PASS 项通过"
    exit 0
fi