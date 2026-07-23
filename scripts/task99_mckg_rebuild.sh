#!/bin/bash
# Task #99 — 重建 MCKG Toys embedding, 强制混合曲率信号
#
# 与原 mckg_M3_c0.5_dim32_toys 对比:
#   - 原: init_kappas=[1.0,0.0,-1.0] → 学到 [+0.84,-0.17,-1.06], 几何信号弱
#   - 新: init_kappas=[5.0,0.0,-5.0] → 强球面/欧氏/强双曲, 几何信号强
#   - κ clamp: 2.0 → 8.0 (允许更大 curvature)
#   - 加 curv_reg_weight=0.1: 鼓励 κ 间距 + norm_cv 匹配 target
#   - target_cvs: [0.30, 0.10, 1.50] (球面紧凑/欧氏中等/双曲分散)
#   - dim: 32 → 64 (几何特征更显式)
#
# 用法: bash scripts/task99_mckg_rebuild.sh

set -e

DATA_DIR="/home/wlia0047/ar57/wenyu/MCKG_repro/MCKG_data/toys"
SAVE_DIR="/home/wlia0047/ar57/wenyu/GeneRec/products/task99_mckg_rebuild"
LOG_DIR="/home/wlia0047/ar57/wenyu/GeneRec/logs/task99_mckg_rebuild"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

cd /home/wlia0047/ar57/wenyu/GeneRec

# cuda:3 空闲 (cuda:0 = Task #87 v6, cuda:1/2 = Task #85 m=0/m=1)
CUDA_VISIBLE_DEVICES=3 python3 task_artifacts/scripts/mckg_model/mckg.py \
    --data_dir "$DATA_DIR" \
    --gpu 0 \
    --M 3 \
    --dim 64 \
    --n_hops 2 \
    --n_neighbors 8 \
    --num_epochs 60 \
    --batch_size 1024 \
    --lr 1e-3 \
    --c 0.5 \
    --eval_every 5 \
    --patience 6 \
    --seed 42 \
    --weight_decay 0.0 \
    --lr_patience 3 \
    --init_kappas "5.0,0.0,-5.0" \
    --kappa_clamp 8.0 \
    --curv_reg_weight 0.1 \
    --target_cvs "0.30,0.10,1.50" \
    2>&1 | tee "$LOG_DIR/train.log"

echo "[task99] 训练完成, 产物落盘到 $SAVE_DIR/entity_embedding.pt"
echo "[task99] 下一步: 重跑 D0 诊断 (scripts/task22_pm_rq_d0_diagnosis.py) 验证混合曲率"