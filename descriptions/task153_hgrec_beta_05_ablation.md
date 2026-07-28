# Task #153 — HG-Rec β=0.5 ablation (paper-faithful commitment loss)

> **任务目的**: 验证 paper Table 6 Instruments 行 β=0.5 是否解释 R@10 -22.4% 差距
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #84 HG-Rec 主实验复现 R@10=0.1020, paper=0.1315, Δ -22.4%.
Task #84 使用 `--beta 1.0` (R11.3 默认值).
paper Table 6 Instruments 行: **β=0.5**.

可能根因之一: commitment loss 权重过大 → codebook 过度 commitment → 表征空间缩小 → R@10 偏低.

---

## 2. 实验设计

**变量**: `--beta 1.0` → `--beta 0.5` (paper-faithful)
**保持不变** (跟 Task #84 baseline 完全一致):
- `--loss_type poincare`, `--num_emb_list 64 128 256`, `--e_dim 32`, `--sk_epsilons 0.0 0.0 0.000`
- `--layers 512 256 128 64`, `--epochs 1000`, `--batch_size 1024`, `--lr 1e-3`
- `--learner AdamW`, `--lr_scheduler_type linear`, `--warmup_epochs 20`
- Stage 2 SID + Stage 3 T5-small + Stage 4 test eval (跟 Task #84 pipeline 一致)

**启动命令**: `scripts/task153_hgrec_beta05_pipeline.sh`

**R7 GPU**: GPU 1 (空闲 — Task #151 P5-CID 占 GPU 0, Task #152 已完成释放 GPU 2)

---

## 3. 决策触发 (vs Task #84 β=1.0)

| Task #153 R@10 | Δ vs Task #84 | Δ vs paper | 解读 |
|----------------|---------------|-----------|------|
| [0.1107, 0.1315] | +8% ~ +29% | paper ±25% 内 | β=0.5 是主要根因 ✅ |
| [0.0986, 0.1107) | -3% ~ +8% | -25% ~ -3% | β 是部分根因, 还有别的主因 |
| [0.0821, 0.0986) | -19% ~ -3% | -37% ~ -25% | β 不是主因, 调查别处 |
| < 0.0821 | < -19% | < -37% | ❌ 异常, 调查 |

---

## 4. 预算

| 阶段 | 估算 |
|------|------|
| Stage 1 RQ-VAE 训练 | ~25 min (1000 epoch) |
| Stage 2 SID | ~3 min |
| Stage 3 T5 训练 (early stop ~75 ep) | ~70 min |
| Stage 4 test eval | ~30 sec |
| **总计** | **~1.7h** |

---

## 5. 风险与缓解

- **codebook 退化**: β 小 → codebook 更新弱 → 利用率可能掉 → 监控 util ≥ 95% 阈值
- **β 太小 commitment loss 失效**: β=0.5 是 paper 值, 已被 paper 验证稳定 → 跟 paper 一致即可

---

## 6. 完成度跟踪

- [ ] Stage 1 RQ-VAE 训练 (β=0.5)
- [ ] Stage 2 SID 落盘
- [ ] Stage 3 T5 训练 + early stop
- [ ] Stage 4 test eval R@10 落盘
- [ ] 写 verdict `verdicts/task153_hgrec_beta05_result.md`
- [ ] 更新 loop.md §16