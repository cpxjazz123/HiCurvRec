# Task #22 Phase 1 — PM-RQ Toy Test Verdict

> **完成日期**: 2026-07-19 02:20
> **状态**: ⚠️ **FAIL — RESTRUCTURE REQUIRED** (Euclidean codebook collapse)
> **下一阶段**: **Phase 1b** — 用 Adam 码本优化器替换 moving average, 重跑 Toy test

---

## 1. 实验目的

验证 Product Manifold RQ (PM-RQ) Toy 实现可行性:
- K=64 三分量 (sphere / euclid / hyperbolic) 单层码本
- 10000 Toys items subset
- 验证 K³ 搜索可行 + 三分量稳定 + Kendall τ 独立性

## 2. 关键指标

| 指标 | 值 | 阈值 | 状态 |
|------|----|----|------|
| **utilization sphere** | **1.000** | > 0.8 | ✅ PASS |
| **utilization euclid** | **0.156** | > 0.8 | ❌ **FAIL (码本坍缩)** |
| **utilization hyperbolic** | **0.875** | > 0.8 | ✅ PASS (边界) |
| Kendall τ (s vs e) | 0.080 | < 0.7 | ✅ PASS |
| Kendall τ (s vs h) | 0.005 | < 0.7 | ✅ PASS |
| Kendall τ (e vs h) | 0.025 | < 0.7 | ✅ PASS |
| search time / item | 0.063 ms | < 50 ms | ✅ PASS |

**D1.5 决策**: **STOP / RESTRUCTURE** (1/3 codebook collapse)

## 3. 详细分析

### 3.1 通过的指标 (6/7)

1. **Kendall τ 三分量独立性**: 全 < 0.1 → 三 κ 子空间**强独立**, 学到不同信号
   - 这与 Task #99 D0 诊断的 `subspace mean cos = -0.17` (强独立) 一致
2. **Search time**: 0.063 ms/item (远低于 50ms 阈值) → K³ 搜索可行
3. **Sphere utilization**: 1.0 (所有 64 个码字都被使用) → 球面几何信号**强**
4. **Hyperbolic utilization**: 0.875 (略低于 0.9 但稳定) → 双曲几何信号**强**
5. **Loss 趋势**: 30.9 → 2.6 (epoch 30) → 19.6 (epoch 60 regress) → 6.5 (epoch 90)

### 3.2 失败的指标 (1/7)

**Euclidean codebook collapse (util = 0.156)**:
- 仅 10/64 码字被使用
- 训练全程 util_e ∈ [0.14, 0.22], 极稳定地坍缩
- 这是 **码本更新机制问题**, 而非 embedding 问题:
  - 用了 `simple moving average` (`Cs[k] = 0.9*Cs[k] + 0.1*mean(batch_used)`)
  - 没有梯度信息, 容易陷入 batch mean 模式
  - 应该用 Adam / RSGD 直接优化重建 loss

### 3.3 重要 insight

**MCKG κ₁ (Euclidean) 子空间本身是健康的**:
- D0 诊断: norm_cv = 4.746 (中等分散)
- Kendall τ = 0.08 (与 sphere/hyperbolic 强独立)
- 子空间编码了与 sphere/hyperbolic 不同的几何信号

**Euclidean 码本坍缩是实现问题**, 不是数据/embedding 问题。

## 4. 根因分析 (Hypothesis)

| 因素 | 评估 |
|------|------|
| MCKG euclidean 子空间分布 | 健康 (D0 norm_cv = 4.746) |
| MCKG euclidean 子空间与 sphere/hyp 独立性 | 强 (Kendall τ = 0.08) |
| MCKG euclidean 子空间信息量 | 与 sphere/hyp 同量级 (loss ≈ 同) |
| **码本更新机制** | **❌ 缺陷: MA → 缺乏梯度信号, batch-mean mode collapse** |
| 码本初始化 (Kaiming randn) | ✅ OK |
| 距离度量 (Poincaré / Euclidean / Cosine) | ✅ OK (geometrically correct) |

## 5. Phase 1b 计划 (RESTRUCTURE)

**目标**: 用 Adam 优化器直接最小化重建 loss, 替换 moving average

**关键改动**:
1. 码本作为 `nn.Parameter`, 启用 autograd
2. Encoder + Decoder 用 nn.Module (从 batch 输入 → 码字最近 → decoder 输出)
3. 重建 loss: `MSE(decoder(quantize(encoder(x))), x)`
4. 三个码本独立 Adam optimizer, lr=1e-3
5. 添加码本 utilization penalty (鼓励码字均衡使用)
6. 添加 batch normalization on input

**预期**: Euclidean util_e 应能恢复到 > 0.7

**启动**: `python3 scripts/task22_pm_rq_phase1b_toy.py --emb-path products/task99_mckg_rebuild/entity_embedding.pt`

**决策**: Phase 1b 通过 → 进入 Phase 2 (K=256, full Toys); Phase 1b 仍 FAIL → 终止 Task #22, 写终局 verdict

## 6. 产物清单

| 产物 | 路径 |
|------|------|
| Toy Test 报告 | `reports/task22_pm_rq/phase1_toy_report.md` |
| Toy metrics JSON | `products/task22_pm_rq/phase1_toy/phase1_metrics.json` |
| 训练日志 | `logs/task22_pm_rq/phase1_toy/train.log` |
| Phase 1 脚本 | `scripts/task22_pm_rq_phase1_toy.py` (备份版) |

## 7. 后续行动

1. **立即**: 实施 Phase 1b (用 Adam optimizer + encoder-decoder), 重跑 Toy test
2. **Phase 1b 通过**: 进入 Phase 2 (Full-Scale Single-Layer K=256)
3. **Phase 1b 仍 FAIL**: 终止 Task #22, 写终局 verdict "PM-RQ Toy 实现无法稳定 Euclidean 码本, 多几何 RQ-VAE 收益有限"

---

**result**: Phase 1 Toy test 6/7 指标 PASS, 但 Euclidean 码本坍缩 (util=0.156) → D1.5 STOP/RESTRUCTURE → 立即 Phase 1b 用 Adam 替换 moving average 重跑.

result: Task #22 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
