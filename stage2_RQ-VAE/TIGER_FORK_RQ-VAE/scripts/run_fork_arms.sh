#!/bin/bash
# Exp2: behaviour-fork-aware representation, hyperbolic grouping vs its matched
# Euclidean control. One arm at a time; Stage2 then Stage3.
set -u
LOCK=/tmp/hicurvrec_fork_driver.lock
exec 9>"$LOCK"
if ! flock -n 9; then echo "another fork driver holds $LOCK" >&2; exit 1; fi

TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_FORK_RQ-VAE
STAGE3=/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train
PY2=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
PY3=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10
LOG=$TREE/logs/fork_driver.log

ARMS="fork_ang_only fork_radial_corr"
mkdir -p "$TREE/logs"
echo "=== fork driver start $(date -Is) ===" >> "$LOG"
for ARM in $ARMS; do
    echo "--- $ARM stage2 start $(date -Is) ---" >> "$LOG"
    sed -i "s/^EXPERIMENT_ARM = \".*\"/EXPERIMENT_ARM = \"$ARM\"/" "$TREE/train_rqvae.py"
    mkdir -p "$TREE/logs/$ARM"
    # Single-threaded: sklearn KMeans reduces in thread order, so this is what
    # makes the codebook init reproducible and shared across arms.
    ( cd "$TREE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $PY2 train_rqvae.py > "logs/train_${ARM}.log" 2>&1 )
    if ! grep -qa 'exported raw_shape' "$TREE/logs/$ARM/train_migrated.log"; then
        echo "--- $ARM STAGE2 FAILED ---" >> "$LOG"; continue
    fi
    echo "--- $ARM stage2 done, stage3 start $(date -Is) ---" >> "$LOG"
    sed -i "s/^ARM = \".*\"/ARM = \"$ARM\"/" "$STAGE3/scripts/run_stage3_fork_arms.py"
    ( cd "$STAGE3" && $PY3 scripts/run_stage3_fork_arms.py > "/tmp/stage3_${ARM}.log" 2>&1 )
    echo "--- $ARM stage3 done $(date -Is) ---" >> "$LOG"
done
echo "=== fork driver end $(date -Is) ===" >> "$LOG"
