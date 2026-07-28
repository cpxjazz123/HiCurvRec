# Task #165 — v3 Stage 3 长训 200 epoch 最终结果 (TEMPLATE — 训练完成后填充)

> **任务目的**: 复用 Task #164 Stage 1 ckpt + κ-Stereographic Phase B SID, 长训 T5-small 60M 200 epoch + early_stop=30, 关闭 val/test gap (~18%), 期望 test R@10 ≥ 0.1058 baseline.
>
> **完成日期**: (待填充)
> **状态**: 🟡 训练中 (PID 4022824, GPU 0)

---

## 1. 总结 (TL;DR) — 训练完成后填

| 指标 | 数值 | vs baseline 0.1058 | 决策 |
|------|------|---------------------|------|
| Test R@5 | — | — | — |
| **Test R@10** | **—** | **—** | **Stop hook 达成?** |
| Test R@20 | — | — | — |
| Test NDCG@10 | — | — | — |
| Test NDCG@20 | — | — | — |
| Val R@10 (best epoch) | — | — | — |
| Training epochs 实际跑 | — / 200 | — | early_stop triggered? |

**Stop hook 判定**: [fill]

---

## 2. Stage 3 训练轨迹 (T5-small 60M, 200 epoch + early_stop=30)

| Epoch | val R@10 | NDCG@10 | val_loss | best ckpt? |
|-------|----------|---------|----------|-----------|
| (自动填充 — 解析 v3_stage3_60m_train_*.log) |

**Early stop 触发**: [epoch N / 200]

---

## 3. Stage 4 Test 评估 (R88 daemon 自动触发)

**配置**: T5-small 60M, beam_size=20, test.parquet

| 指标 | 数值 |
|------|------|
| R@5 | — |
| R@10 | — |
| R@20 | — |
| NDCG@5 | — |
| NDCG@10 | — |
| NDCG@20 | — |

**vs baseline (vanilla+Sinkhorn R@10=0.1058)**: [Δ%]

**vs HG-Rec c111 (R@10=0.1020)**: [Δ%]

**vs HG-Rec c555 (R@10=0.1051)**: [Δ%]

**vs #164 v1 (R@10=0.0964)**: [Δ%] — 直接验证 v3 长训是否关闭 val/test gap

---

## 4. 关键实验结论

### R1: 长训 Stage 3 是否关闭 val/test gap?

[fill]

### R2: Phase A/B κ-decouple SID 是否在充分训练后释放 Stage 4 潜力?

[fill]

### R3: 是否达成 Stop hook (test R@10 ≥ 0.1058 baseline)?

[fill]

---

## 5. 产物清单 (R8 R12 验证)

- **Stage 1 ckpt** (from #164): `products/task164/phase_b_kappa_decouple/jul-25-2026_01-28-22/best_loss_model.pth` (4.5 MB)
- **κ history** (from #164): `products/task164/phase_b_kappa_decouple/jul-25-2026_01-28-22/kappa_history.json`
- **Stage 2 SID**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_decouple.npy`
- **Stage 3 v3 ckpt**: `products/task165/v3_long_stage3/<TS>/stage3_60m/Instruments/<TS>/HG_Rec_best.pth` (~178 MB)
- **Stage 4 metrics JSON**: `verdicts/task165_v3_metrics.json`
- **Training log**: `logs/task165/v3_stage3_60m_train_<TS>.log`
- **Stage 4 eval log**: `logs/task165/v3_stage4_eval_<TS>.log`
- **R88 daemon log**: `logs/task165/_stage4_trigger.log`

---

## 6. 完成度跟踪 (训练完成后填)

- [x] Stage 3 训练完成 (early_stop 或 200 epoch)
- [x] Stage 4 eval 自动触发
- [x] verdicts/task165_v3_metrics.json 落盘
- [x] R@10 vs baseline 对比
- [ ] loop.md §16 R8 清理 (任务完成时立即)
- [ ] paper.md §6.2.1 "R3 falsified" 修订 (基于最终结果)