#!/bin/bash
# Task #164 κ-Stereographic short training test (200 epochs)
# Validates: κ-Stereographic formula trains without NaN on real Instruments data
# Uses same params as Task #162 paper-faithful (β=0.5, [32,64,256], sk=0)
# Plus θ_init=[0, 0, 0] (κ_init=0 everywhere, no ORC bias) to isolate formula correctness

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task164
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/kappa_stereo_test_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task164/kappa_stereographic_test/${TS}
mkdir -p $LOG_DIR $PROD_DIR

PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #164 κ-Stereographic short test] launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: κ-Stereographic distance formula + θ_init=[0,0,0] (κ_init=0) + paper β=0.5 + [32,64,256] + sk=0" | tee -a $LOG_FILE
echo "GPU 2 (CUDA_VISIBLE_DEVICES=2, completely idle)" | tee -a $LOG_FILE
echo "Test target: 200 epochs short run, verify no NaN + κ convergence trajectory" | tee -a $LOG_FILE
echo "vs #163 Phase D: 1000 epochs θ_init=[-0.0114, +0.2375, +0.2375] (NaN @ ep 764)" | tee -a $LOG_FILE
echo "vs #162 paper-free-curv: same θ_init=[0,0,0] but old 3-branch torch.where formula" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=2 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 5e-3 \
    --theta_init_list 0.0 0.0 0.0 \
    --kappa_max 2.0 \
    --seed 42 \
    --num_emb_list 32 64 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 0.5 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --dead_code_reset_every 0 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $PROD_DIR \
    --kappa_log_path $PROD_DIR/kappa_history.json \
    --log_interval 20 \
    --save_every 100 \
    2>&1 | tee -a $LOG_FILE

echo "===== [Task #164 κ-Stereographic short test] completed at $(date) =====" | tee -a $LOG_FILE
rm -f $PID_FILE