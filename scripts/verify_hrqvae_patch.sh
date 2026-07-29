#!/bin/bash
# Issue #18 Gate 0 (a) — 校验 hrqvae_trainer.py line 322 + line 4 状态
# 2026-07-29
#
# 通过条件 (Issue #18 Gate 0 (a) 单条规则):
#   - Line 322 = "import glob" (单 import, 不再带 "import os")
#   - Line 4 = "import os" (顶层 import 保留)
#
# 用法: bash scripts/verify_hrqvae_patch.sh

set -e
REPO=$(cd $(dirname $0)/.. && pwd)
TARGET=$REPO/HG-Rec/model/hrqvae_trainer.py

if [ ! -f "$TARGET" ]; then
    echo "❌ FAIL: $TARGET not found (新 clone 未 apply patches)"
    echo "   请先: bash scripts/apply_hgrec_patches.sh"
    exit 1
fi

LINE_322=$(sed -n '322p' "$TARGET")
LINE_4=$(sed -n '4p' "$TARGET")

echo "Line 4   = '$LINE_4' (期望: 'import os')"
echo "Line 322 = '$LINE_322' (期望: 'import glob')"

RET=0
# Trim 前后空白比较 (行内有缩进)
LINE_4_TRIM=$(echo "$LINE_4" | xargs)
LINE_322_TRIM=$(echo "$LINE_322" | xargs)
[ "$LINE_4_TRIM" = "import os" ] || { echo "❌ Line 4 (trim) ≠ 'import os' (got: '$LINE_4_TRIM')"; RET=1; }
[ "$LINE_322_TRIM" = "import glob" ] || { echo "❌ Line 322 (trim) ≠ 'import glob' (got: '$LINE_322_TRIM')"; RET=1; }

# Optional: 自动 locate smoke run hrqvae.log (在 $OUT_DIR 子目录), 验证 Gate 0 (b) 残留 (--check-smoke)
if [ "${1:-}" = "--check-smoke" ]; then
    SMOKE_DIR=${SMOKE_DIR:-$REPO/products/issue18_gate0_smoke}
    HRQVAE_LOG=$(find "$SMOKE_DIR" -name "hrqvae.log" 2>/dev/null | head -1)
    if [ -z "$HRQVAE_LOG" ]; then
        echo "❌ --check-smoke: $SMOKE_DIR 下无 hrqvae.log"
        exit 1
    fi
    echo "Smoke log: $HRQVAE_LOG"
    STEP2_COUNT=$(grep -c "\[step2 monitor ep" "$HRQVAE_LOG" 2>/dev/null | head -1)
    STEP2_COUNT=${STEP2_COUNT:-0}
    echo "Step2 monitor lines: $STEP2_COUNT (期望 ≥ 1)"
    [ "$STEP2_COUNT" -ge 1 ] 2>/dev/null || { echo "❌ step2 monitor 未打印"; exit 1; }
    UNBOUND_COUNT=$(grep -c "UnboundLocalError\|referenced before assignment" "$HRQVAE_LOG" 2>/dev/null | head -1)
    UNBOUND_COUNT=${UNBOUND_COUNT:-0}
    echo "UnboundLocalError count: $UNBOUND_COUNT (期望 = 0)"
    [ "$UNBOUND_COUNT" -eq 0 ] 2>/dev/null || { echo "❌ 仍有 UnboundLocalError"; exit 1; }
    grep "step2 monitor ep1" "$HRQVAE_LOG" | head -1
fi

if [ "$RET" -eq 0 ]; then
    echo "✅ Issue #18 Gate 0 (a) PASS: hrqvae_trainer.py line 4 + line 322 状态正确"
    echo "   (Issue #17 Gate 1 修复已 apply, 新 clone 可重现)"
fi
exit $RET
