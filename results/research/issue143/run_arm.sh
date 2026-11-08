#!/bin/bash
# One arm, one card, detached from the caller's process group.
arm="$1"; card="$2"
d="/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_$arm"
cd "$d" || exit 1
GENREC_SINGLE_GPU=1 CUDA_VISIBLE_DEVICES="$card" PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  setsid nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_$arm/train_rqvae.py" \
  > "$d/logs/train_card${card}.log" 2>&1 < /dev/null &
echo "$arm pid=$! card=$card"
