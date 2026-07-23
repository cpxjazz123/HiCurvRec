#!/bin/bash
# Task #87: Stage 2.2 → Stage 3 → Stage 4 流水线 launcher
#
# 前提: Stage 2.1 (rqvae_train_tiger) 已训练完毕, ckpt 在 logs/task87_s2_rqvae_train_v6/.../checkpoints/ 下
# 用法: bash scripts/task87_pipeline_part2.sh <stage2_ckpt_path>
#   e.g.: bash scripts/task87_pipeline_part2.sh logs/task87_s2_rqvae_train_v6/runs/task87_s2_train/checkpoints/checkpoint_*.ckpt
#
# Steps:
#   1. Stage 2.2 inference (rqvae_inference_tiger) → cluster_ids.pt (3, 11924)
#   2. Bridge: build dedup col → (4, 11924) tensor
#   3. Stage 3 train (tiger_train_tiger) on cuda:3 (~6-8h)
#   4. Stage 4 inference on cuda:0
#   5. Stage 4 evaluation → verdicts/task87_tiger_baseline_eval.json

set -e

CKPT="${1:?Usage: $0 <stage2_ckpt_path>}"
EMB_PATH="logs/task87_s1_sentence_t5_base_inference/runs/task87_s1/pickle/merged_predictions_tensor.pt"
DATA_DIR="data/amazon_data/toys"

source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# === Stage 2.2: RQ-VAE inference (cuda:0) ===
S2_INFER_DIR="logs/task87_s2_inference_tiger/runs"
CUDA_VISIBLE_DEVICES=0 python3 -m src.inference experiment=rqvae_inference_tiger \
    data_dir="$DATA_DIR" \
    embedding_path="$EMB_PATH" \
    embedding_dim=768 num_hierarchies=3 codebook_width=256 \
    ckpt_path="$CKPT" \
    id=task87_s2_infer \
    task_name=task87_s2_rqvae_inference_v6

S2_CLUSTER_PT=$(ls -t $S2_INFER_DIR/task87_s2_infer/pickle/cluster_ids.pt | head -1)
echo "[pipeline] Stage 2.2 done -> $S2_CLUSTER_PT"
if [ ! -f "$S2_CLUSTER_PT" ]; then
    echo "[pipeline] ERROR: cluster_ids.pt not found" >&2; exit 1
fi

# === Bridge: build dedup col → (4, 11924) tensor ===
SID4_PATH="products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt"
mkdir -p "$(dirname $SID4_PATH)"
python3 scripts/task15_stage22_to_stage3_bridge.py \
    --input "$S2_CLUSTER_PT" \
    --output "$SID4_PATH" \
    --n-catalog 11924
echo "[pipeline] Bridge done -> $SID4_PATH ($(python3 -c "import torch;print(torch.load('$SID4_PATH',weights_only=True).shape)"))"

# === Stage 3: TIGER train (cuda:3, ~6-8h) ===
echo "[pipeline] Launching Stage 3 TIGER train (cuda:3, ~6-8h) ..."
CUDA_VISIBLE_DEVICES=3 nohup python3 -m src.train experiment=tiger_train_tiger \
    data_dir="$DATA_DIR" \
    semantic_id_path="$SID4_PATH" \
    num_hierarchies=4 \
    sequence_length=120 \
    id=task87_s3_train \
    task_name=task87_s3_tiger_train > logs/task87_s3_train.log 2>&1 &
S3_PID=$!
echo "[pipeline] Stage 3 PID=$S3_PID"

# 监控 Stage 3 进度（每 5 min 检查一下 val/recall@5）
echo "[pipeline] Will monitor Stage 3 every ~5 min. Logs at logs/task87_s3_train.log"
echo "[pipeline] Stage 3 checkpoint (best val/recall@5) will land in:"
echo "[pipeline] logs/task87_s3_tiger_train/runs/task87_s3_train/checkpoints/"
