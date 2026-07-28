#!/bin/bash
set -e
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

CKPT_DIR="/home/wlia0047/ar57/wenyu/GeneRec/products/task188/hrqvae_save_limit50"
SCRIPT="/home/wlia0047/ar57/wenyu/GeneRec/scripts/task189_codebook_geometry_diagnose.py"
LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs/task189/time_series"
mkdir -p $LOG_DIR

# 用 find 找 epoch_*_collision_*.pth (避开 globbing [64,...])
# 对每个 epoch 列出 ckpt
declare -a EPOCHS=(54 109 184 354 414 659 794)
for ep in "${EPOCHS[@]}"; do
    CKPT=$(find "$CKPT_DIR" -name "epoch_${ep}_collision_*.pth" 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "Skip epoch=$ep (no ckpt)"
        continue
    fi
    COLL=$(basename "$CKPT" | grep -oE "collision_[0-9.]+" | sed 's/collision_//')
    echo "Epoch $ep collision=$COLL  ckpt=$CKPT"
    cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
    python3 -u "$SCRIPT" \
        --ckpt_path "$CKPT" \
        --label "Task188_EPOCH${ep}_coll${COLL}" \
        --device cuda:0 \
        > "${LOG_DIR}/epoch_${ep}.out" 2>&1
done

echo "DONE — Task #189 time-series (7 epochs)"
