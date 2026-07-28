# Task #200 Phase 0 — 双码本几何解耦 初始化验证 (v3 通过)

> **任务目的**: 验证用户给的"球面 kmeans 方向码本 + latent 均值重构码本"初始化算法在三层上是否产生可用码本
>
> **完成日期**: 2026-07-26
> **状态**: ✅ **PASS** (v3 L0 中心化后三层全 PASS)

---

## 1. 任务目标

承接 #195-#199 系列: 在不改双曲几何 loss 的前提下,把分配半径从残差幅度的副产品变为显式设计变量。用户 2026-07-26 给定方案:
- `emb_geo`: 球面 k-means 方向聚类, 决定分配索引
- `emb_rec`: latent 的 per-cluster 均值, 决定重构幅度
- L0 在 encoder 输出 z 上跑(因 z 不以原点为中心),L1/L2 在残差上跑

Phase 0 任务:**不训练, 只验证 init_emb 是否产生可用码本**(4 个指标 + 数值安全)。

---

## 2. 执行时间线

| 版本 | 时间 | 修改 | 结果 |
|------|------|------|------|
| v1 | 00:50 | util=/N, cos_max<0.95 硬约束 | 三层全 FAIL |
| v2 | 01:10 | util=/K, cos_max 改 info, 加 cone_ratio 诊断 | L1/L2 PASS, **L0 FAIL (cos_mean=+0.27)** |
| v3 | 01:20 | **L0 加 z - mean(z) 中心化**, L1/L2 不动 | **三层全 PASS** |

**脚本**: `scripts/task200_phase0_dual_codebook_init.py` (CPU only, 不污染上游)
**日志**:
- v2 (无中心化): `logs/task200/phase0_metrics_centerFalse.json`
- v3 (L0 中心化): `logs/task200/phase0_metrics_centerTrue.json`

---

## 3. 关键指标 (v3, L0 中心化)

| Layer | cos_mean | cos_max (info) | n_dup | min ‖residual‖ | util (=1) | 判定 |
|-------|----------|----------------|-------|----------------|-----------|------|
| **L0** | **+0.0004** ✅ | 0.9932 | 0 ✅ | 0.0413 ✅ | **1.0** ✅ | **PASS** |
| L1 | +0.0235 ✅ | 0.9720 | 0 ✅ | 0.0300 ✅ | 1.0 ✅ | PASS |
| L2 | +0.0426 ✅ | 0.9657 | 0 ✅ | 0.0180 ✅ | 1.0 ✅ | PASS |

**Phase 0 通过标准** (用户 2026-07-26 修正):
- `cos_mean ∈ [-0.05, 0.15]` ✓
- `n_dup(cos>0.995) = 0` ✓
- `util (unique SID / K) = 1.0` ✓ (修正后: util 三层全 1.0)
- `min ‖residual‖ > 1e-6` ✓

---

## 4. 锥体紧致度诊断 (用户 2026-07-26 要求, 印证设计洞察)

| Layer | cone_ratio = ‖mean(z)‖ / mean(‖z‖) | pw_cos_mean (1000 pairs) | 含义 |
|-------|--------------------------------------|---------------------------|------|
| **L0** | **0.5064** | **+0.2514** | z 在远离原点的紧锥里 → **必须中心化** |
| L1 | 0.1321 | +0.0204 | 残差天然中心化, 健康 |
| L2 | 0.1738 | +0.0351 | 同上 |

**核心洞察** (来自用户):
- L0 输入 = encoder 输出 z = μ + δ_i, ‖μ‖=0.155 远大于 mean(‖δ‖)≈0.15
- 归一化后所有方向 ≈ μ/‖μ‖ → 挤在一个锥里 (cos_mean 0.27)
- L1/L2 输入是**残差** r = z - e_L0, 天然均值 ≈ 0, 不需要中心化

**修复**: L0 加 z = z - mean(z) 中心化 (Phase 1 训练时改用 EMA). cos_mean 从 +0.27 → **+0.0004**, 完全消除锥体效应。

---

## 5. v1 → v2 修正 (util 算法错误)

**v1 错误** (我之前报的): util = 0.65% / 1.29% / 2.58% → 误判"坍缩到几个簇"

**真相**: 报的数字 = K/N = 64/9922, **0.645%**, 不是利用率。当 K 个码字全部用上时, 这个数字 = K/N, 跟 util 无关。

**用户原文**: "util 的计算是 (用到的码字数) / (item 数), 而不是 / 码本大小。当所有码字都被用上时, 这个数就恰好等于 K/N."

**修复**: 改为 `len(assign.unique()) / K`. 三层立即全 1.0.

---

## 6. 与原 v1 verdict 的更正

**v1 我说的**:
- ❌ "sklearn kmeans 在 32-dim 严重坍缩到几个簇"
- ❌ "L0 失败是初始化算法结构性失败, 需换球面 kmeans / Sobol init"

**真相**:
- ✅ v1 的 util 数据 (0.65%/1.29%/2.58%) 完全等于 K/N, 反而**证明码字全部用上**
- ✅ L0 失败**只是** z 不以原点为中心 (cone_ratio=0.51), 单点修复 z=z-mean(z) 解决
- ✅ 球面 kmeans 替换不是必须, 反而可能引入新问题(质心归一化的几何偏差)

---

## 7. 决策触发 → Phase 1

**主判据 (Phase 0 层面)**:
| 检查 | 通过标准 | v3 实测 |
|------|----------|---------|
| 几何激活 | ρ = 2r 恒成立 | (Phase 2 验证) |
| **没有方向坍缩** | cos_mean ∈ [-0.05, 0.15], n_dup=0 | ✓ 三层全在区间 |
| 码本可用 | util=1.0, collision ≤ 15% | ✓ util=1.0 (collision 在训练时算) |
| 数值稳定 | 无 NaN, min ‖r‖ > 1e-6 | ✓ 三层全 > 0.018 |

**结论**: Phase 0 **PASS**, 进入 Phase 1 (冒烟 50 epoch).

---

## 8. 风险与缓解 (Phase 1 之前)

| 风险 | 缓解 |
|------|------|
| 中心化在训练时需要 EMA | 用 `z_mean.mul(0.99).add(latent.mean(0), alpha=0.01)`, eval 时固定 |
| 中心化改变了 encoder 输出语义 | encoder gradient 仍回流到原 z, 中心化只是码本初始化, 不参与反传 |
| 三层都中心化会更安全 | 用户决定只 L0, L1/L2 不动 (残差天然中心化) — 遵守 |
| min ‖r‖ 偏低 (L2=0.018) | 用户原文提到"如果接近 0 先解决", 0.018 远大于 1e-6, 安全 |

---

## 9. 后续动作

1. **进入 Phase 1**: 单臂 50 epoch 冒烟 (用户指定臂 C, ρ=2.0/2.7/3.4 → r_target = 1.0/1.35/1.70)
2. **判据**: 无 NaN, cos_max 不上升, collision < 30%
3. **Phase 2**: 4 臂 (B/C/D/E) × 1000 epoch (主推 C)

---

## 10. 产物清单

- [scripts/task200_phase0_dual_codebook_init.py](../../scripts/task200_phase0_dual_codebook_init.py) — 支持 `--center_l0` 开关, cone_ratio 诊断, util 修正
- [logs/task200/phase0_metrics_centerFalse.json](../../logs/task200/phase0_metrics_centerFalse.json) — v2 (无中心化): L0 fail, L1/L2 pass
- [logs/task200/phase0_metrics_centerTrue.json](../../logs/task200/phase0_metrics_centerTrue.json) — v3 (L0 中心化): 三层全 PASS
- 本 verdict 文件

---

**result: Phase 0 双码本几何解耦 init ✅ PASS — L0 加 z=z-mean(z) 中心化后三层全过 (util=1.0, cos_mean ∈ [-0.05, 0.15], n_dup=0, min‖r‖ > 0.018). 锥体诊断证实用户洞察: L0 cone_ratio=0.51 需中心化, L1/L2 <0.18 天然中心化. 下一步: Phase 1 单臂 50 epoch 冒烟.**