#!/usr/bin/env bash
# Task launcher with SID swap (R10 + R11.5 + R7 safe)
# Usage: launch_with_sid_swap.sh <sid_target_path> <gpu_id> <arm> <task_id> <log_dir> <output_dir>

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
SID_TARGET="$1"
GPU_ID="$2"
ARM="$3"
TASK_ID="$4"
LOG_DIR="$5"
OUTPUT_DIR="$6"

# Validate args
if [ -z "$SID_TARGET" ] || [ -z "$GPU_ID" ] || [ -z "$ARM" ]; then
    echo "Usage: $0 <sid_target_path> <gpu_id> <arm> <task_id> <log_dir> <output_dir>"
    exit 1
fi

if [ ! -f "$SID_TARGET" ]; then
    echo "❌ SID file not found: $SID_TARGET"
    exit 1
fi

# Backup original + swap symlink
SID_HARDCODED="$REPO/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue30_per_layer_transforms.npy"
BACKUP_PATH="$REPO/HG-Rec/dataset/Instruments/.backup_sid_swap_${TASK_ID}_${ARM}.npy"

cp "$SID_HARDCODED" "$BACKUP_PATH"
ln -sf "$SID_TARGET" "$SID_HARDCODED"

# Validate swap (use realpath for symlink resolution)
ACTUAL_TARGET=$(realpath "$SID_HARDCODED" 2>/dev/null || readlink -f "$SID_HARDCODED")
SID_TARGET_REAL=$(realpath "$SID_TARGET" 2>/dev/null || readlink -f "$SID_TARGET")
if [ "$ACTUAL_TARGET" != "$SID_TARGET_REAL" ]; then
    echo "❌ SID swap failed: $SID_HARDCODED → $ACTUAL_TARGET (expected $SID_TARGET_REAL)"
    cp "$BACKUP_PATH" "$SID_HARDCODED"
    rm -f "$BACKUP_PATH"
    exit 1
fi

echo "✅ SID swapped: $SID_HARDCODED → $SID_TARGET (SHA256 of target = $(md5sum $SID_TARGET | cut -d' ' -f1))"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/stage3_${ARM}_${TS}.log"

echo "===== [Task #${TASK_ID} Arm ${ARM}] launched at $TS =====" | tee "$LOG"
echo "GPU: $GPU_ID, SID target: $SID_TARGET" | tee -a "$LOG"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG"

PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python
export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" $REPO/scripts/task320_issue38_5arm_stage3_train.py \
    --arm "$ARM" --num_epochs 200 --batch_size 256 --gpu 0 --task_id "$TASK_ID" 2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}

# Restore original SID
cp "$BACKUP_PATH" "$SID_HARDCODED"
rm -f "$BACKUP_PATH"

ACTUAL_RESTORED=$(md5sum "$SID_HARDCODED" | cut -d' ' -f1)
ORIGINAL_RESTORE_HASH=$(md5sum $REPO/HG-Rec/dataset/Instruments/.backup_issue30_sid_20260730_2359.npy | cut -d' ' -f1)

if [ "$ACTUAL_RESTORED" != "$ORIGINAL_RESTORE_HASH" ]; then
    echo "❌ SID restore FAILED. Manual restore needed: cp $REPO/HG-Rec/dataset/Instruments/.backup_issue30_sid_20260730_2359.npy $SID_HARDCODED"
    exit 1
fi

echo "✅ SID restored to original. Exit code: $EXIT_CODE" | tee -a "$LOG"
echo "===== [Task #${TASK_ID} Arm ${ARM}] finished at $(date +%Y%m%d_%H%M%S) =====" | tee -a "$LOG"

exit $EXIT_CODE