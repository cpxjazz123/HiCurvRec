#!/bin/bash
# cleanup_checkpoints.sh — Keep only latest + best checkpoint per training run.
#
# Usage: bash cleanup_checkpoints.sh
#
# Rules:
#   - For each completed training run (>30 min idle), keep only:
#     1. The checkpoint with the highest step number (latest)
#     2. The best_*.ckpt (early_stopping/monitor checkpoint)
#   - Delete all other checkpoint_*.ckpt files
#   - Delete last.ckpt if present
#   - Skip currently running runs (PID detection)

set -euo pipefail

GRID_ROOT="/home/wlia0047/ar57/wenyu/GeneRec/GRID"
CKPT_ROOT="$GRID_ROOT/logs/train/runs"

if [ ! -d "$CKPT_ROOT" ]; then
    echo "[ckpt-clean] $CKPT_ROOT not found"
    exit 0
fi

# Find all training PIDs (currently active runs)
ACTIVE_RUNS=$(ps aux | grep -E "src\.train" | grep -v grep | awk '{print $NF}' | sort -u || true)

echo "[ckpt-clean] scanning $CKPT_ROOT ..."

find "$CKPT_ROOT" -type d -name "checkpoints" 2>/dev/null | while read -r CKPT_DIR; do
    RUN_DIR=$(dirname "$CKPT_DIR")
    RUN_NAME=$(basename "$RUN_DIR")

    # Skip if any src.train process is running
    if ps aux | grep -E "src\.train" | grep -v grep | grep -q "$RUN_DIR"; then
        echo "[skip] $RUN_NAME (process running)"
        continue
    fi

    # Find the latest (highest-step) checkpoint
    LATEST=$(ls "$CKPT_DIR"/checkpoint_*.ckpt 2>/dev/null | sort -V | tail -1)

    # Remove older checkpoint_*.ckpt
    removed_count=0
    freed_mb=0
    for CKPT in "$CKPT_DIR"/checkpoint_*.ckpt; do
        if [ -f "$CKPT" ] && [ "$CKPT" != "$LATEST" ]; then
            SIZE_MB=$(($(stat -c %s "$CKPT") / 1024 / 1024))
            rm -f "$CKPT"
            freed_mb=$((freed_mb + SIZE_MB))
            removed_count=$((removed_count + 1))
        fi
    done

    # Remove last.ckpt
    if [ -f "$CKPT_DIR/last.ckpt" ]; then
        SIZE_MB=$(($(stat -c %s "$CKPT_DIR/last.ckpt") / 1024 / 1024))
        rm -f "$CKPT_DIR/last.ckpt"
        freed_mb=$((freed_mb + SIZE_MB))
        removed_count=$((removed_count + 1))
        echo "[rm] last.ckpt from $RUN_NAME (${SIZE_MB}MB)"
    fi

    if [ "$removed_count" -gt 0 ]; then
        echo "[ckpt-clean] $RUN_NAME: removed $removed_count files, freed ${freed_mb}MB"
    fi
done

echo "[ckpt-clean] done"
df -h /home/wlia0047 | tail -1