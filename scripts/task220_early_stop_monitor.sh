#!/bin/bash
# Task #220 早期中止 monitor wrapper
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

python3 -u $REPO/scripts/task220_early_stop_monitor.py \
    --task_id 220 \
    --log_file $REPO/logs/task220/pck_stage1_train.out \
    --pid_file $REPO/products/task220/_TRAINING_PID \
    --ckpt_dir $REPO/products/task220/hrqvae_pck \
    --check_epochs 20 50 \
    --util_kill_below 0.30 0.50 \
    --collision_kill_above 0.95 0.80 \
    --assignment_mode per_codeword_kappa \
    --c_k_min 0.5 --c_k_max 5.0