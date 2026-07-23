# Task #77 — LETTER t5-base 训练 (backbone 升级)

> **任务目的**: LETTER backbone t5-small → t5-base. Task #50/61 训练 t5-small 已达 R@10=0.0997 (超 paper 0.0581, +72%). t5-base 模型容量更大, 期待 R@10 > 0.12.
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

LETTER-TIGER Task #50/61 复现成功, R@10=0.0997, 用 `google/t5-small` backbone (~60M 参数). 升级到 `google/t5-base` (~220M 参数) 可能显著提升生成质量.

**注意**: RQ-VAE tokenizer (Instruments.index.epoch5000.alpha0.01-beta0.0001.json, 692 tokens) **沿用 Task #50**, 训练 tokenizer 已对齐 Musical_Instruments 24587 items.

## 2. 实验设计

**变量**: backbone t5-small → **t5-base**
**保持不变**: 
- LETTER 框架 + Trie constrained beam search
- RQ-VAE tokenizer (692 tokens)
- Instruments.index.epoch5000.alpha0.01-beta0.0001.json
- learning_rate=5e-4, epochs=200, per_device_batch_size=256, temperature=1.0
- early_stopping patience=20 (与 Task #50 一致)

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER

# 单卡 (GPU 3), 不用 DDP
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=disabled

DATASET=Instruments
OUTPUT_DIR=./ckpt/${DATASET}_t5base/

python3 finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --base_model google/t5-base \
    --per_device_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.epoch5000.alpha0.01-beta0.0001.json \
    --temperature 1.0 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/data
```

> per_device_batch_size 128 → 64 (t5-base 是 t5-small 的 3.7× 容量, 显存压力大)

## 3. 决策阈值

| R@10 | 决策 |
|------|------|
| ≥ 0.12 | ⭐⭐ t5-base 突破, 写 paper extension |
| 0.10-0.12 | ✅ t5-base 与 t5-small 等价 |
| 0.08-0.10 | 🟡 t5-base 过拟合 |
| < 0.08 | ❌ t5-base 退化, 保持 t5-small |

## 4. 预算

~4-6h (t5-base 是 t5-small 3.7×, 训练时间 ~1.5×)

## 5. 风险

- 显存: t5-base + bs=64 ≈ 30-40 GB on L40S, 需监控 OOM
- 收敛: t5-base 可能过拟合, 监控 val_loss 趋势, early_stop=20 patience
- 路径冲突: ckpt 路径用 `${DATASET}_t5base/` 后缀避免覆盖 Task #50 ckpt