# Task #150 — LETTER paper-aligned 修复 (降低 over-train + 恢复 paper defaults)

> **任务目的**: LETTER paper-aligned 复现 R@10=0.0997 vs paper 0.0581 (+71.6% 过正). 根因是 task72 / task136 / task139 launcher 使用了 **论文 25-50× 过大的训练超参** (lr=5e-4 vs paper 2e-5, batch=256 vs paper 8, epochs=200 vs paper default 4). 修复: 切回 paper default 并验证 R@10 落在 [0.052, 0.075] 区间 (±25% paper).

> **完成日期**: in progress
> **状态**: 🟡 待启动

---

## 1. 背景 (用户 2026-07-24 反馈)

用户审查 baseline Table 2 时指出:
> "我们是希望对齐paper的方法, 而不是直接去使用新的方法. 帮我修复- LETTER +- P5-CID"

**LETTER +71.6% 过正根因**:
- `LETTER/LETTER-TIGER/finetune.py` paper default: `lr=2e-5, batch=8, epochs=4, temperature=1.0`
- 当前 launcher (`scripts/task139_launch_letter_tiger.sh` + `scripts/task72_phase3_letter_train.sh`): `lr=5e-4, batch=256, epochs=200`
- 我们的 launcher 用 paper 设置的 **25× lr + 32× batch + 50× epoch**, 让 T5-small 在 rich title 上 memorize 训练集 → test 时 R@10 over-shoot
- **修复**: 切回 paper 的小 lr 小 batch 短 epoch; 用 paper-faithful setup 重训 LETTER-TIGER.

## 2. 实验设计

**变量** (3 个超参切回 paper default):
- `--learning_rate 5e-4 → 2e-5`
- `--per_device_batch_size 256 → 8`
- `--epochs 200 → 4` (paper default, 加 early stopping 到 patience=20 防止 under-training)

**保持不变**:
- 框架: LETTER/LETTER-TIGER/finetune.py
- GPU: 0 或 2 (R7 空闲优先, 不抢 Task #149 GPU 1)
- Multi-GPU: `torchrun --nproc_per_node=2` (跟 paper 一致)
- 数据集: Instruments
- index_file: `.index.json`
- 其它 flag (weight_decay, warmup_ratio) 保持 RecBole-style 现有 default

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/LETTER/LETTER-TIGER
CUDA_VISIBLE_DEVICES=0 \
  torchrun --nproc_per_node=2 --master_port=2315 finetune.py \
    --output_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task150/ckpt_paper_aligned \
    --dataset Instruments \
    --per_device_batch_size 8 \
    --learning_rate 2e-5 \
    --epochs 4 \
    --index_file .index.json \
    --temperature 1.0 \
    --seed 42 \
    --early_stopping_patience 20
```

## 3. 决策触发 (vs paper R@10=0.0581)

| Task #150 test R@10 | Δ vs paper | 决策 |
|---------------------|------------|------|
| [0.052, 0.064] (-10% ~ +10%) | ✅ paper-aligned | 闭环成功, 写 verdict + 更新 Table 2 |
| [0.064, 0.075] (+10% ~ +30%) | ⚠️ marginal | 微调 epoch=6 或 temperature=0.7 重跑 |
| [0.040, 0.052] (-30% ~ -10%) | ⚠️ under-train | lr=5e-5 试一次 (paper 默认 + half) |
| > 0.075 或 < 0.040 | ❌ 异常 | 调查数据 / prompt format |

→ **目标区间**: R@10 ∈ [0.052, 0.075] (paper 0.0581 ±25%).

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| Paper-aligned 训练 4 epoch | ~30 min (batch 8 × 2 GPU ≈ paper 原始时序) | GPU 0 |
| Test eval | ~5 min | GPU 0 |
| **总计** | **~35 min** | GPU 0 |

## 5. 风险与缓解

**风险 1**: 4 epoch 太短可能 under-train.
**缓解**: EarlyStopping callback 已配置 patience=20, 4 epoch 跑完后 test R@10 不达标 → 切 epoch=8 重跑.

**风险 2**: batch 8 vs paper batch 8 一致, 数据并行 2 卡, 全球 batch=16 → 比 paper 的 8 大 2×.
**缓解**: 若 R@10 偏 paper 默认 0.10+ 边界外, 把 `per_device_batch_size=8` → `4` 让 global=8 完全对齐 paper.

**风险 3**: tokenizer 路径或 .index.json 路径差异.
**缓解**: 沿用 task139 现有 setup, 不改 tokenize 步骤.

## 6. 完成度跟踪

- [ ] scripts/task150_letter_paper_aligned.sh 写完
- [ ] 启动到 GPU 0 (R7 空闲优先)
- [ ] 4 epoch training 收敛 + early stop
- [ ] Test eval 完成 R@5/R@10/R@20
- [ ] 写 verdict: `verdicts/task150_letter_paper_aligned_result.md`
- [ ] 更新 §16 (R8 旧清理, 当前活跃任务替换)
- [ ] 把 LETTER R@10=0.0997 → 新值, 替换 Table 2 GENERATIVE_BASELINES 行
