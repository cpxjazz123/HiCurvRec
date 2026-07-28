#!/bin/bash
# Task #163 Phase D — paper-faithful HG-Rec free-curv with REAL ORC θ_init
# θ_init = [-0.0114, +0.2375, +0.2375] (computed from validated Lin 2011 W_1 LP)
# 跟 #162 (random init) 和 Phase C (proxy init [-0.316, -0.800, -0.800]) 形成 3-way 对照

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task163
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/phase_d_stage1_train_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task163/phase_d_rqvae_real_orc/${TS}
mkdir -p $LOG_DIR $PROD_DIR

PID_FILE=$PROD_DIR/_TRAINING_PID

echo "===== [Task #163 Phase D Stage 1] REAL ORC θ_init paper-faithful HG-Rec free-curv launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: paper-faithful β=0.5 + [32,64,256] + sk=0 + REAL ORC-derived θ_init [-0.0114, +0.2375, +0.2375]" | tee -a $LOG_FILE
echo "GPU 3 (CUDA_VISIBLE_DEVICES=3, completely idle)" | tee -a $LOG_FILE
echo "θ_init per-layer (computed from validated Lin 2011 W_1 LP on Instruments item-item graph):" | tee -a $LOG_FILE
echo "  L0 = -0.0114 (L0 ORC mean=-0.0228, near-zero → Euclidean init)" | tee -a $LOG_FILE
echo "  L1 = +0.2375 (L0×L1 prefix ORC mean=+0.466, spherical)" | tee -a $LOG_FILE
echo "  L2 = +0.2375 (L2 skipped per user risk #1, use L1 placeholder)" | tee -a $LOG_FILE
echo "  → κ_init = [-0.0228, +0.4662, +0.4662] (κ_max=2.0)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE
echo "vs Task #162 random init: κ=[+0.167, -0.555, -0.268] → 训练终值 κ=[+1.146, +0.655, +0.745] spherical" | tee -a $LOG_FILE
echo "vs Task #163 Phase C proxy: κ=[-0.612, -1.328, -1.328] (hyperbolic, proxy 已废)" | tee -a $LOG_FILE
echo "vs Task #163 Phase D (本次): κ=[-0.023, +0.466, +0.466] (Euclidean L0, spherical L1/L2)" | tee -a $LOG_FILE
echo "Stage 1 fork: scripts/task89_stage1_train_rqvae.py" | tee -a $LOG_FILE

CUDA_VISIBLE_DEVICES=3 python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 1000 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 5e-3 \
    --theta_init_list -0.0114 +0.2375 +0.2375 \
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

echo "===== [Task #163 Phase D Stage 1] Training completed at $(date) =====" | tee -a $LOG_FILE
rm -f $PID_FILE