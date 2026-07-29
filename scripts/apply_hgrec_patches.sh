#!/bin/bash
# HG-Rec patches 一键 apply 脚本 (idempotent)
# 2026-07-29 (Task #274)
#
# 解决问题: .gitignore 排除 HG-Rec/ 整个目录, 任何 HG-Rec/ 文件修复都是 disk-only local fix.
#          新 clone 不会自动获得这些修复.
#          本脚本 + patches/*.patch 提供 patch 文件化方案, 不修改 .gitignore 也不 fork 上游.
#
# 用法: bash scripts/apply_hgrec_patches.sh

set -euo pipefail
REPO=$(cd $(dirname $0)/.. && pwd)
PATCH_DIR=$REPO/patches
TARGET_DIR=$REPO/HG-Rec/model

if [ ! -d "$PATCH_DIR" ]; then
    echo "❌ patches directory not found: $PATCH_DIR"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ HG-Rec/model directory not found: $TARGET_DIR"
    echo "   请先 git clone 或 hg clone 上游 HG-Rec repo 到 $REPO/HG-Rec/"
    exit 1
fi

APPLIED=0
SKIPPED=0
FAILED=0

for patch_file in "$PATCH_DIR"/*.patch; do
    [ -e "$patch_file" ] || continue  # empty glob

    patch_name=$(basename "$patch_file")
    echo "--- 处理 patch: $patch_name ---"

    # Forward dry-run test
    if patch -d "$REPO" --dry-run -p1 < "$patch_file" > /tmp/patch_dry_fwd.out 2>&1; then
        # Apply forward
        if patch -d "$REPO" -p1 < "$patch_file" > /tmp/patch_apply.out 2>&1; then
            echo "  ✅ APPLIED: $patch_name"
            APPLIED=$((APPLIED+1))
        else
            echo "  ❌ APPLY FAILED: $patch_name"
            cat /tmp/patch_apply.out
            FAILED=$((FAILED+1))
        fi
        continue
    fi

    # Forward dry-run failed. Check if it's because already applied.
    # patch 命令 'Reversed (or previously applied) patch detected' 是关键 signal.
    if grep -qE "previously applied|Reversed" /tmp/patch_dry_fwd.out 2>/dev/null; then
        # Verify via reverse dry-run (反向 dry-run 成功 → 正向已 apply)
        if patch -d "$REPO" --dry-run -R -p1 < "$patch_file" > /tmp/patch_dry_rev.out 2>&1; then
            echo "  ⏭️  SKIPPED (already applied): $patch_name"
            SKIPPED=$((SKIPPED+1))
        else
            echo "  ❌ REVERSE DRY-RUN FAILED: $patch_name"
            cat /tmp/patch_dry_rev.out
            FAILED=$((FAILED+1))
        fi
    else
        echo "  ❌ DRY-RUN FAILED: $patch_name"
        cat /tmp/patch_dry_fwd.out
        FAILED=$((FAILED+1))
    fi
done

echo "---"
echo "✅ Total: applied=$APPLIED skipped=$SKIPPED failed=$FAILED"
exit $FAILED
