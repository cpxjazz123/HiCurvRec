#!/usr/bin/env bash
# P4: 3-seed 复现验证选点规则稳定性
#
# 选点规则 (写死 - 见 选点规则.md):
#   - 数据: P3 user-disjoint split (13509 train / 2898 val / 3005 test)
#   - 验证: diag_val - 唯一允许监控的指标是 val_R@10_Generative (autoregressive, next_k=4)
#   - 测试: diag_test - 全程封存, 只最后 eval 1 次
#   - 停止: 连续 3 次评估 (间隔 500 步) val_R@10_Generative 不提升 (Δ < 0.001)
#   - 选 ckpt: best val_R@10_Generative 对应的 checkpoint
#   - 不允许: 看 test 调整、改 min_delta / patience、训练后挑点
#
# 接受标准 (P4 final):
#   - 3 个 seed 选出的 step 范围 ≤ 1500 步
#   - test_R@10 (3 seeds) 标准差 ≤ 0.005

set -e

DATA_DIR=/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys
SID_PATH=/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task19_aq_s22/pickle/merged_predictions_tensor.pt
GRID=/home/wlia0047/ar57/wenyu/GeneRec/GRID

OUT_BASE=/home/wlia0047_ar57_scratch/wenyu/p4_3seed || true
OUT_BASE=${OUT_BASE:-/home/wlia0047/ar57_scratch/wenyu/p4_3seed}

# 3 个 seeds
SEEDS=(42 43 44)

mkdir -p ${OUT_BASE}

# 训练 cmdline overrides (写死, 不允许改)
TRAIN_OVERRIDES=(
    "experiment=tiger_decoder_only_train_flat"
    "data_dir=${DATA_DIR}"
    "semantic_id_path=${SID_PATH}"
    "num_hierarchies=4"
    "sequence_length=120"
    "ckpt_path=null"
    "data_loading.train_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_train"
    "data_loading.val_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_val"
    "callbacks.model_checkpoint.monitor=val/recall@10"
    "callbacks.model_checkpoint.save_top_k=1"
    "callbacks.early_stopping.monitor=val/recall@10"
    "callbacks.early_stopping.patience=3"
    "callbacks.early_stopping.min_delta=0.001"
    "callbacks.early_stopping.mode=max"
    "trainer.val_check_interval=500"
    "trainer.max_steps=10000"
    "trainer.accelerator=gpu"
    "trainer.devices=1"
    "paths.root_dir=${GRID}"
)

for SEED in "${SEEDS[@]}"; do
    SEED_DIR=${OUT_BASE}/seed_${SEED}
    mkdir -p ${SEED_DIR}

    echo "=========================================="
    echo "Seed ${SEED}: training"
    echo "=========================================="

    if [ ! -f "${SEED_DIR}/last.ckpt" ] && [ ! -f "${SEED_DIR}/best.ckpt" ]; then
        source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
        conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
        cd ${GRID}
        PYTHONPATH=${GRID} python -m src.train \
            "${TRAIN_OVERRIDES[@]}" \
            "seed=${SEED}" \
            "paths.log_path=${SEED_DIR}" \
            2>&1 | tee "${SEED_DIR}/train.log"
    else
        echo "Seed ${SEED}: 已完成, 跳过训练"
    fi

    # Find best ckpt by val/recall@10 prefix
    BEST_CKPT=$(ls ${SEED_DIR}/checkpoints/checkpoint_*_val_recall@10*.ckpt 2>/dev/null | sort -t= -k4 -n | tail -1)
    if [ -z "${BEST_CKPT}" ]; then
        echo "Seed ${SEED}: 没找到 best ckpt, 跳过 test eval"
        continue
    fi

    echo "=========================================="
    echo "Seed ${SEED}: test eval (only ONCE on diag_test)"
    echo "  ckpt: ${BEST_CKPT}"
    echo "=========================================="
    cd ${GRID}
    PYTHONPATH=${GRID} python -m src.inference \
        experiment=tiger_inference_flat \
        data_dir=${DATA_DIR} \
        semantic_id_path=${SID_PATH} \
        ckpt_path=${BEST_CKPT} \
        num_hierarchies=4 \
        sequence_length=120 \
        "paths.root_dir=${GRID}" \
        "paths.output_path=${SEED_DIR}/test_eval" \
        "data_loading.predict_dataloader_config.dataloader.data_folder=${DATA_DIR}/diag_test" \
        2>&1 | tee "${SEED_DIR}/test_eval.log"
done

echo "=========================================="
echo "P4 done. Now aggregate into summary"
echo "=========================================="
cd ${GRID}
PYTHONPATH=${GRID} python3 task_artifacts/scripts/p4_aggregate.py \
    --seed_dirs ${OUT_BASE}/seed_42 ${OUT_BASE}/seed_43 ${OUT_BASE}/seed_44 \
    --out ${OUT_BASE}/p4_summary.json
