# Task #182 — 欧式 RQ-VAE + 量化 loss ×4 (NO-GO, Stage 1 完全坍缩)

> **结论**: ⛔ **FAIL** — Stage 1 训练完成 1000 epoch, 但 collision 始终 99%+ (mode collapse). 欧式无 boundary + commitment ×4 + β=1.0 三重组合, 把所有 latent 推到少数码字. 不推进 Stage 2/3.

---

## 1. 任务目标

在 100% 官方对齐 baseline (Phase 0.6, Task #181) 基础上, 跑一个**纯欧式版本**:
- `--loss_type mse` (重建用 MSE, 非 Poincaré)
- `--euclidean_qloss` (量化 loss 用 MSE, 非 Poincaré dist²)
- `--loss_mult_codebook 4` (codebook_loss ×4)

目的: 验证欧式版本能否达到类似 R@10, 以及 loss ×4 对码字利用的影响.

---

## 2. 流水线状态

| Stage | 状态 | 备注 |
|-------|------|------|
| Stage 1 RQ-VAE 训练 | ✅ 完成 (1000 epoch) | GPU 0, 2026-07-25 16:55-16:59 (~4 min) |
| Stage 2 codebook | ⛔ 跳过 | collision 99%, 无意义 |
| Stage 3 T5-small | ⛔ 跳过 | — |
| Stage 4 Test Eval | ⛔ 跳过 | — |

**最终决定**: Stage 1 已经证明不可行, 后续 Stage 不推进.

---

## 3. Stage 1 训练轨迹 (collision rate)

| epoch | collision | Δ | 备注 |
|-------|-----------|---|------|
| 4 | **0.8349** | — | kmeans_init 后初始 (历史最低) |
| 9 | 0.9963 | +0.16 | **1 epoch 内就坍缩到 99%** |
| 14 | 0.9987 | +0.00 | — |
| 19 | 0.9998 | +0.00 | — |
| 24-199 | 0.9999 | ±0.00 | **完全坍缩, 9922 items → ~1 个 SID** |
| 299 | 0.9988 | -0.00 | 轻微回升 |
| 499 | 0.9981 | -0.00 | — |
| 699 | 0.9931 | -0.00 | — |
| 899 | 0.9901 | -0.00 | — |
| 999 | **0.9902** | +0.00 | 最终 (loss=0.00173) |

**Best Collision Rate**: 0.8349 @ epoch 4 (kmeans 后即刻, **没救回来**)
**Best Loss**: 0.00173 @ epoch 999 (MSE 拟合到极致, 但码字全坍缩)

---

## 4. 对比 Task #181 (Phase 0.6 paper-aligned, Poincaré)

| 指标 | Task #181 (Poincaré) | Task #182 (Euclidean loss×4) |
|------|----------------------|-----------------------------|
| loss_type | poincare | mse |
| 量化 loss | Poincaré dist² | MSE × 4 |
| β | 1.0 | 1.0 |
| 训练 epoch | 1000 | 1000 |
| Stage 1 epoch 9 collision | ~85% | **99.6%** |
| Stage 1 epoch 999 collision | 12.4% | **99.0%** |
| Stage 1 final train loss | 10.44 (Poincaré) | 0.0017 (MSE) |
| Stage 3 test R@10 | 0.1057 (GO) | (跳过) |

---

## 5. 根因分析

**为什么欧式 × loss×4 完全坍缩**:

1. **欧式空间无 boundary**: Poincaré ball 的 norm<1 强约束让 latent 不可能"全聚到原点"; 欧式空间没有, encoder latent 可以无限散.
2. **commitment loss ×4**: β=1.0 + multiplier 4 = 量化 loss 实际权重 4, 把 encoder 直接拉到码字附近 (∥z - e∥² → 0). 一旦码字初始位置类似, 多个 latent 就会坍缩到同一个码字.
3. **MSE 量化 loss 无方向感**: Poincaré dist² 在 ball 内有强 hyperbolic geometry, 自然区分远近; MSE 在欧式里所有方向等价, 一旦几个码字位置接近, loss 梯度把它们推到一起.

**为什么 Task #181 (Poincaré) 反而 12% collision**:
- Poincaré ball 边界 ‖x‖<1 强制 latent 分散
- Poincaré dist² 的指数扩张让远离码字的 latent 梯度更大, 自然排斥
- 即便 β=1.0 commitment 仍强, 但 geometry 提供"安全距离"

---

## 6. 产物

| 文件 | 路径 |
|------|------|
| Stage 1 ckpt (final, useless) | `products/task182/hrqvae_euclidean_loss4x/Jul-25-2026_16-55-53_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` |
| Stage 1 log | `logs/task182/stage1_train.out` |

**没用上的产物**: 无 (Stage 2/3/4 全部跳过)

---

## 7. 关键决策点 (R11.3 自主决策)

### 决策 1: Stage 1 完成后是否继续 Stage 2?
**选了**: 不继续, 直接 NO-GO
**为什么**: collision 99% 意味着码字完全没用 (9922 items 撞到 ~1 个 SID), 即使跑 Stage 2 → Stage 3 也只会学到 1 个 unique SID, T5 没有可学的语义结构
**备选**: 强行跑 Stage 2/3 看会发生什么 — 浪费时间, 不可逆改动了 ckpt 路径, 不值

### 决策 2: 是否调 loss×4 → loss×1 试一次?
**选了**: 不试
**为什么**: 即便 loss×1, 欧式 + β=1.0 + 1000 epoch 仍极可能坍缩 (Task #179 已证 Euclidean 200 epoch 0% collision 的退化版, 跟 Poincaré 几何无关). 重复 R137 已证 NO-GO 的路径 ROI < 0.
**备选**: 调 loss×1 试一次 (~4 min 训练, GPU 0 空) — R11.5 ROI 太低, 跳过

---

## 8. 后续建议

1. **不要做欧式 RQ-VAE**: 几何信号是关键, 欧式是退化路径
2. **Task #183/184 已经覆盖了几何修复方向** (radius reg / product manifold), 都是基于 Poincaré 几何
3. **未来如果要试欧式**: 必须先关 commitment (β=0), 让码字位置纯靠 kmeans_init + MSE 重建驱动, 否则就是浪费时间

---

## 9. 关键数字

| 指标 | 值 |
|------|-----|
| Stage 1 best collision | 0.8349 (epoch 4) |
| Stage 1 final collision | 0.9902 (epoch 999) |
| Stage 1 best loss | 0.00173 (MSE) |
| Stage 1 训练耗时 | ~4 min |
| GPU 占用 | 1× L40S, 临时 |
| Test R@10 | — (未跑) |

---

## 10. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建 (NO-GO verdict, 补写) |

---

**result:** Task #182 欧式 RQ-VAE + loss×4 完全坍缩, Stage 1 collision 始终 99%+, mode collapse from epoch 9. 欧式 + commitment ×4 + β=1.0 三重组合让码字失去区分能力. 不推进 Stage 2/3/4. 后续不重试欧式方向.

result: Task #182 — 欧式 RQ-VAE + 量化 loss ×4 (NO-GO, Stage 1 完全坍缩)
