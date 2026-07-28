#!/bin/bash
# Task #222 早停 monitor — ep30 检查 utilization + collision
set -e
REPO=/home/wlia0047/ar57/wenyu/GeneRec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd $REPO

export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models

python3 -u $REPO/scripts/task220_early_stop_monitor.py \
    --task_id 222 \
    --log_file $REPO/logs/task222/pck_replay_stage1_train.out \
    --pid_file $REPO/products/task222/_TRAINING_PID \
    --ckpt_dir $REPO/products/task222/hrqvae_pck_replay \
    --check_epochs 30 \
    --util_kill_below 0.15 \
    --collision_kill_above 0.50 \
    --assignment_mode per_codeword_kappa \
    --c_k_min 0.5 --c_k_max 5.0