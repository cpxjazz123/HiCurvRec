#!/bin/bash
# Run the three Exp4 arms end to end: Stage2 then Stage3, one arm at a time.
#
# Each arm differs only in the behaviour-term geometry, so the sequence is:
#   fix_euclid_raw   rebuilt baseline, raw latent, plain Euclidean
#   fix_euclid_ball  same fixed ball map, Euclidean metric  -> isolates the map
#   fix_poincare     same fixed ball map, Poincare metric   -> isolates the metric
#
# The arm constant is rewritten in place before each run, so whatever is on disk
# when a run starts is exactly what that run used. The Stage3 launcher is pointed
# at the same arm so its CODE_PATH and output dirs line up.
set -u

# Single-instance guard. Two drivers racing for the GPU silently produced
# half-finished arms and inconsistent artifacts; a lock makes that impossible.
LOCK=/tmp/hicurvrec_exp4_driver.lock
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another Exp4 driver already holds $LOCK; refusing to start" >&2
    exit 1
fi

TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_HYPERBOLIC_RQ-VAE
STAGE3=/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train
PY2=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9
PY3=/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10
DRIVER_LOG=$TREE/logs/exp4_driver.log

ARMS="fix_det_euclid_raw fix_det_euclid_ball fix_det_poincare fix_carry_euclid_raw"

mkdir -p "$TREE/logs"
echo "=== Exp4 driver start $(date -Is) ===" >> "$DRIVER_LOG"

for ARM in $ARMS; do
    echo "--- $ARM stage2 start $(date -Is) ---" >> "$DRIVER_LOG"
    sed -i "s/^EXPERIMENT_ARM = \".*\"/EXPERIMENT_ARM = \"$ARM\"/" "$TREE/train_rqvae.py"
    mkdir -p "$TREE/logs/$ARM"
    # Single-threaded BLAS/OMP: sklearn's KMeans reduces in thread-order, so a
    # multi-threaded run gives a slightly different codebook every launch
    # (~1e-5 relative) even at a fixed seed. Single-threading makes the codebook
    # bit-identical across arms and across retries, so the three Exp4 arms
    # provably start from the same quantization, and it also avoids the
    # degenerate L3 collapse that killed two arms on the first attempt.
    ( cd "$TREE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $PY2 train_rqvae.py > "logs/train_${ARM}.log" 2>&1 )
    if ! grep -qa 'exported raw_shape' "$TREE/logs/$ARM/train_migrated.log"; then
        echo "--- $ARM STAGE2 FAILED, skipping stage3 ---" >> "$DRIVER_LOG"
        continue
    fi
    echo "--- $ARM stage2 done, stage3 start $(date -Is) ---" >> "$DRIVER_LOG"
    sed -i "s/^ARM = \".*\"/ARM = \"$ARM\"/" "$STAGE3/scripts/run_stage3_hyperbolic_arms.py"
    ( cd "$STAGE3" && $PY3 scripts/run_stage3_hyperbolic_arms.py > "/tmp/stage3_${ARM}.log" 2>&1 )
    echo "--- $ARM stage3 done $(date -Is) ---" >> "$DRIVER_LOG"
done

echo "=== Exp4 driver end $(date -Is) ===" >> "$DRIVER_LOG"
