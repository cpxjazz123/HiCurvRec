#!/bin/bash
# Task #174 Stage 1 — κ-Stereographic + M=2 + 门控网络融合 (用户 design §5 D 臂)
# 跟 #170/#171/#172 对照. 假设门控机制 (gating) 在 κ_m=0 附近仍能学出有意义合成,
# 避免 Task #89 A 臂 (M=1 κ_m=0 NO-GO) 的失效模式.
# GPU 选择: GPU 0 (R7: 等 #170 Stage 3 完成后立即 launch) 或 GPU 1/2/3

set -e

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

LOG_DIR=/home/wlia0047/ar57/wenyu/GeneRec/logs/task174
mkdir -p $LOG_DIR
TS=$(date +%b-%d-%Y_%H-%M-%S | sed 's/^.../\L&/')
LOG_FILE=$LOG_DIR/stage1_phase_ab_${TS}.log

PROD_DIR=/home/wlia0047/ar57/wenyu/GeneRec/products/task174/phase_b_mckg_gating/${TS}
mkdir -p $PROD_DIR

export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task174_s1
mkdir -p "$TRITON_CACHE_DIR"

echo "===== [Task #174 Stage 1] κ-Stereo + M=2 + 门控融合 D 臂 launched at $(date) =====" | tee $LOG_FILE
echo "Recipe: κ-Stereographic + θ_init=[0,0,0] + Phase A 100ep frozen + Phase B 100ep lr_theta=1e-5 + M=2 + gating" | tee -a $LOG_FILE
echo "D 臂: dist(x,c) = Σ_m w_m(x) · d_{κ_m}(x_m, c_m), w_m = softmax(MLP(x))_m" | tee -a $LOG_FILE
echo "vs #170 (κ-Stereo + Sinkhorn ALL 3 layers): M=1, no gating" | tee -a $LOG_FILE
echo "vs #172 (κ_max=4.0): M=1, no gating" | tee -a $LOG_FILE
echo "vs A 臂 Task #89: M=1, no gating → NO-GO κ_m=0" | tee -a $LOG_FILE
echo "Target: test R@10 > 0.1058 (Stop hook 条件 C3)" | tee -a $LOG_FILE
echo "Output: $PROD_DIR" | tee -a $LOG_FILE

# 关键改动: --M 2 + 用 task174 launcher (MCKGGatingHRQVAE)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TRITON_CACHE_DIR="$TRITON_CACHE_DIR" python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task174_stage1_train_mckg_gating.py \
    --M 2 \
    --gating_enabled \
    --gate_hidden 32 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 1e-5 \
    --theta_init_list 0.0 0.0 0.0 \
    --kappa_max 2.0 \
    --kappa_freeze_epochs 100 \
    --seed 42 \
    --num_emb_list 32 64 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 0.5 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --device cuda:0 \
    --ckpt_dir $PROD_DIR \
    --log_interval 10 \
    --save_every 200 \
    --kappa_log_path $PROD_DIR/kappa_history.json \
    --phase_a_baseline_util_path $PROD_DIR/phase_a_baseline_util.json \
    2>&1 | tee -a $LOG_FILE

STAGE1_EXIT=${PIPESTATUS[0]}
echo "===== [Task #174 Stage 1] exit code: $STAGE1_EXIT at $(date) =====" | tee -a $LOG_FILE

if [ $STAGE1_EXIT -ne 0 ]; then
    echo "❌ Stage 1 FAILED" | tee -a $LOG_FILE
    exit 1
fi

echo "✅ Task #174 Stage 1 完成" | tee -a $LOG_FILE
ls -la $PROD_DIR/best_loss_model.pth | tee -a $LOG_FILE