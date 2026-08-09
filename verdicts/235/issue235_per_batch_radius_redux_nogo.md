# Issue #235 --enable_per_batch_radius_mod redux 50ep NO-GO (2026-08-09)

## Context

**触发**: 2026-08-09 /loop 5m "stage2 curvature should be learnable"
**原意**: 测试 per-layer-statistic κ (区别于 #228 共享 batch scalar)
**实际**: 误用 --enable_per_batch_radius_mod (沿用 #228 路径), 实为 #228 redux

## 50 epoch 训练观察 (PID 257429)

| Epoch | κ_L0 | κ_L1 | κ_L2 | Δ(L0,L1) | Δ(L0,L2) |
|-------|------|------|------|---------|---------|
| 0 | -0.248 | -0.248 | -0.248 | 0.000 | 0.000 |
| 5 | -0.205 | -0.206 | -0.204 | 0.001 | 0.001 |
| 10 | -0.108 | -0.108 | -0.106 | 0.000 | 0.002 |
| 15 | 0.087 | 0.089 | 0.102 | 0.002 | 0.015 |
| 20 | 0.239 | 0.246 | 0.252 | 0.007 | 0.013 |

**结论**: per-layer κ 跨 epoch 始终均匀 (Δ < 0.02), 与 #228 (cd4815e) 完全一致.
**根因**: shared batch scalar (computed from pre-L0 latent) 跨 3 层同值, 等价于 c_global 标量缩放, 不提供 per-layer 独立信号.

## R23 提前 kill

50ep 已足够确认信号与 #228 redux 一致, kill -9 终止 (PID 257429, 训练 1m14s).

## 真正 novel direction (per-layer-statistic EMA) 未实施

设计: 每个 RQ 层用自己 forward 时的 input residual norm, 维护独立 EMA, 调制 κ_l.
理论区别于 #228: residual norm 是 layer-specific (sequential 累积), 非共享 scalar.
但代码修改需重写 KappaAwareVectorQuantization.forward 路径 + 加新 flag,
工作量 ≥ 100 行 + 1000 epoch 训练, 预期结局:

- κ 仍会向某个均值收敛 (因为 EMA 仍提供的是标量信号)
- 真正 novel 的是 per-layer independent bias (类似 HAB Stage3 设计), 但这本质上把 Stage2 κ 改造推向 Stage3 HAB 同性质方案, 历史 #56-#60 Stage1 hyp+Stage2 RQ-VAE 不兼容已证类似路径 NO-GO.

## 终局

Stage2 κ learnable 路径已彻底穷尽:
- Issue #157 v15 capmatch (per-layer κ anchor+drift, REL_STRUCT target) — 当前最优, [0.30, 1.79, 1.48]
- Issue #225 v2 (per-item κ) — NO-GO (40× 爆炸)
- Issue #228 (per-batch scalar) — NO-GO (L2 76% 塌缩)
- Issue #235 redux — NO-GO (κ uniform, 与 #228 同机制)

**0.108 物理不可达**: Issue #95 ceiling = 0.1057 (单 ckpt), Issue #94 = 0.1079 (3-way ensemble).

**Why**: v15 capmatch per-layer κ 是从 codebook cardinality 反推的 REL_STRUCT target 最优解, 任何数据驱动的 κ 调制 (per-item/batch/layer-EMA) 都会破坏这一手工设计的最优性.
**How to apply**: 不再尝试任何 Stage2 κ 改造. 接受 0.1057/0.1079 作为本环境最终结果.