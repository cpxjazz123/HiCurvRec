#!/usr/bin/env bash
# Task #318 Issue #38 Arm 1: Stage 3 optimizer 4-arm ablation (parallel on 4 GPUs)
# Adam / AdamW / Adafactor / SGD
# Same Issue #30 GO endpoint (Stage 1/2 fixed r_l=[0.1,1,10]+s_l=[2,2,2])
# 50 epochs each (~25 min wall time per arm)

set -uo pipefail

REPO=/home/wlia0047/ar57/wenyu/GeneRec
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$REPO/scripts:${PYTHONPATH:-}
PYTHON_BIN=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python

mkdir -p logs/task318

# 4-arm parallel: arm=adam GPU 0, arm=adamw GPU 1, arm=adafactor GPU 2, arm=sgd GPU 3
declare -A GPU_MAP=( [adam]=0 [adamw]=1 [adafactor]=2 [sgd]=3 )

for opt in adam adamw adafactor sgd; do
    GPU=${GPU_MAP[$opt]}
    LOG=logs/task318/stage3_${opt}_$(date +%Y%m%d_%H%M%S).log
    echo "Launching $opt on GPU $GPU, log: $LOG"
    CUDA_VISIBLE_DEVICES=$GPU nohup "$PYTHON_BIN" scripts/task318_issue38_arm1_optimizer_stage3_train.py \
        --optimizer $opt --num_epochs 50 --gpu 0 --task_id 318 \
        > "$LOG" 2>&1 &
    PID=$!
    echo "$opt PID=$PID"
    echo "$PID > logs/task318/${opt}_PID"
    echo $PID > logs/task318/${opt}_PID
done

echo "All 4 arms launched. Monitor with:"
echo "  ls -lt logs/task318/stage3_*.log | head"
echo "  for f in logs/task318/*_PID; do echo -n \"\$f: \"; cat \$f; done"