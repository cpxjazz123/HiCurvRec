# Task #154 — HG-Rec 200 epoch ablation (paper-faithful training duration)

> **任务目的**: 验证 paper Table 6 Instruments 行 200 epoch 是否解释 R@10 -22.4% 差距
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #84 HG-Rec 主实验复现 R@10=0.1020, paper=0.1315, Δ -22.4%.
Task #84 用了 `--epochs 1000` (R11.3 默认值, 可能 over-train).
paper Table 6 Instruments 行: **200 epoch, 无 early stop**.

可能根因之一: 训练过久 → 模型 overfit → val 改善但 test 不变 → 实际泛化能力未提升.

---

## 2. 实验设计

**变量**: `--epochs 1000` → `--epochs 200` (paper-faithful)
**保持不变** (跟 Task #84 baseline 完全一致):
- `--loss_type poincare`, `--num_emb_list 64 128 256`, `--e_dim 32`, `--sk_epsilons 0.0 0.0 0.000`
- `--layers 512 256 128 64`, `--beta 1.0`, `--batch_size 1024`, `--lr 1e-3`
- `--learner AdamW`, `--lr_scheduler_type linear`, `--warmup_epochs 20`
- Stage 2/3/4 pipeline 完全一致

**启动命令**: `scripts/task154_hgrec_200epoch_pipeline.sh`

**R7 GPU**: GPU 2 (空闲)

---

## 3. 决策触发 (vs Task #84 epochs=1000)

| Task #154 R@10 | Δ vs Task #84 | Δ vs paper | 解读 |
|----------------|---------------|-----------|------|
| [0.1107, 0.1315] | +8% ~ +29% | paper ±25% 内 | 200 epoch 是主要根因 ✅ |
| [0.0986, 0.1107) | -3% ~ +8% | -25% ~ -3% | 训练时长是部分根因 |
| [0.0821, 0.0986) | -19% ~ -3% | -37% ~ -25% | 不是主因 |
| < 0.0821 | < -19% | < -37% | ❌ 异常 |

---

## 4. 预算

| 阶段 | 估算 |
|------|------|
| Stage 1 RQ-VAE 训练 | ~10 min (200 epoch vs 1000) |
| Stage 2 SID | ~3 min |
| Stage 3 T5 训练 (early stop ~75 ep) | ~70 min |
| Stage 4 test eval | ~30 sec |
| **总计** | **~1.5h** |

---

## 5. 风险

- **codebook 没收敛**: 200 epoch 太少 → best_loss 偏高 → R@10 下降 → 需要监控 best_loss 是否 < 10
- **early stop 触发太早**: patience=20 @ 200 epoch 总共 → 实际可能 ~150 epoch 触发

---

## 6. 完成度跟踪

- [ ] Stage 1 RQ-VAE 训练 (epochs=200)
- [ ] Stage 2 SID 落盘
- [ ] Stage 3 T5 训练 + early stop
- [ ] Stage 4 test eval R@10 落盘
- [ ] 写 verdict `verdicts/task154_hgrec_200epoch_result.md`
- [ ] 更新 loop.md §16