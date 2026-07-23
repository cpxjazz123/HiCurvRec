#!/bin/bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec
export PYTHONUNBUFFERED=1
exec python3 -u scripts/task6_train_rqvae.py "$@"
