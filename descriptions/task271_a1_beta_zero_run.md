# Task #271 — A1 β=0.0 Stage 1 验证 FAIL: L0 utilization 不可达 ≥ 90%

## 背景

Task #270 设计了 3 配方 L0 utilization curriculum (A1/A2/A3). Task #271 实际启动 A1 (β=0.0 纯欧氏 VQ-VAE), 验证假设 "双曲 commit loss 是 L0 坍缩根因".

## 任务范围

1. 启动 A1 launcher (`RECIPE=A1 bash scripts/task270_utilization_curriculum.sh`)
2. 50 epoch × 6s/ep = ~5 min Stage 1, GPU 0
3. 通过条件: L0 ≥ 58/64 (90.625%) @ ep 30, collision < 0.95
4. 失败硬停: 任一条件不满足 → 记 verdict

## 实测结果 (FAIL)

| ep | L0 utilization | L1 | L2 | collision | 备注 |
|----|---------------|----|----|-----------|------|
| 5 | 4.7% (3/64) | 0.8% (1/128) | 2.3% (6/256) | 0.9989 | 严重坍缩早期 |
| 10 | 17.2% (11/64) | 16.4% (21/128) | 9.0% (23/256) | 0.9779 | 上升期 |
| 15 | 21.9% (14/64) | 11.7% (15/128) | 8.6% (22/256) | 0.9889 | 振荡 |
| 20 | **29.7% (19/64)** ↑ peak | 12.5% (16/128) | 9.4% (24/256) | 0.9862 | L0 峰值 |
| 25 | 26.6% (17/64) ↓ | 20.3% (26/128) | 5.5% (14/256) | 0.9869 | 饱和下降 |
| 30 | 26.6% (17/64) | 23.4% (30/128) | 5.9% (15/256) | 0.9869 | **20% 硬 kill @ ep 30** |

**hypnorm 不变**: κ=1.000 ‖x‖_E mean L0=0.076, L1=0.005, L2=0.003, fill 率 7.6% / 0.5% / 0.3% — **码字 Euclidean 范数 50 epoch 内不变** = 码字没从初始 K-means 位置扩散出去

## 关键发现 (比预期更深)

1. **β 不是 L0 坍缩根因**: Task #253 β=0.5 baseline L0=73.44% @ ep50, Task #271 A1 β=0.0 L0=29.7% (peak) @ ep20 / 26.6% (饱和) @ ep30. **β=0 反而更糟**. 假设 "β=0.5 双曲 commit loss 是瓶颈" 证伪.

2. **码字 Euclidean 范数 stuck**: hypnorm 显示 50 epoch 内 ‖x‖_E mean 不变 (L0=0.076). 这是 product_manifold 下 euc 分支的码字范数. **码字不扩散** = encoder 输出也锁定在 K-means 初始化的球面附近.

3. **K=64 容量疑似不足**: 9922 items 分布在 K=64 codes 上, ep20 峰值 L0=29.7% (19 unique codes). 即便监督 (用 dead_revive), K=64 也不够 9922 distinct mode.

4. **20% 硬 kill 触发了**: hrqvae_trainer.py 行 489-502. 这是 Task #265 之前的旧逻辑 — 之前 50 epoch 没打过 utilization 所以从未触发, 现在 step2 monitor 真打印后首次发现 baseline 也会触发.

## 通过条件判定 (FAIL)

| 条件 | 阈值 | 实测 | 判定 |
|------|------|------|------|
| L0 ≥ 58/64 @ ep ≥ 30 | ≥ 90.625% | 26.6% (17/64) | ❌ FAIL (差 63.7pp) |
| collision < 0.95 | < 0.95 | 0.9869 | ❌ FAIL (差 3.7pp) |
| recon_loss ≤ 1500 | ≤ 1500 | ~12 | ✅ PASS |

**总判定**: FAIL. A1 路径不可行.

## 后续 (R11.3)

A1 FAIL 不代表 curriculum 思路全 FAIL, 但**否定 "β 是 L0 bottleneck" 假设**. 后续路径:

| 候选 | 通过概率 | 决策 |
|---|---|---|
| **A3 freeze encoder** (β=0.5 + freeze encoder @ ep 20) | 中 (Task #193 历史已用, 但 product_manifold 下效果待测) | 候选 |
| **A2 curriculum** (A1 warm-start β=0.5) | 低 (A1 起点 L0=26.6% 已不好, warm-start 到 0.5 难反弹) | **弃** |
| **新方向: --num_emb_list 128 256 512** (扩 L0 容量) | 高 (但改 Stage 3 SID 维度, 破坏 Stage 3 compat) | **架构变更, 不在本候选范围** |
| **新方向: encoder variance debugging** (encoder 输出是否真的多模?) | 未知 (需诊断) | 候选 backlog |

**决策**: 标 A1 配方 NO-GO. A2 (curriculum) 因依赖 A1 也 NO-GO. A3 (freeze encoder) 候选保留但 ROI 显著降低 (Task #193 历史已用, 同样 L0 < 90% 风险高).

**建议资源转**: backlog 候选 2 (m-arm κ-Stereographic v9+) 或 候选 4 (VERDICT 库存盘点) — L0 ≥ 90% 在 baseline recipe 下不可达, 是 structural 结论.

## 物理产物

```
logs/task270/stage1_A1_2026-07-29_10-11-02.log  (完整 stdout)
products/task270/A1_euclidean/Jul-29-2026_10-11-33_beta_0.000_codebook_[64,128,256]_sk_0.000/
  - best_collision_model.pth
  - best_loss_model.pth  
  - epoch_9_collision_0.9779_model.pth
  - epoch_24_collision_0.9869_model.pth (last)
descriptions/task271_a1_beta_zero_run.md  (本文件)
verdicts/task271_a1_beta_zero_run_result.md
```

result: Task #271 — A1 β=0.0 FAIL. L0 利用率 ep20 峰值 29.7% / ep30 饱和 26.6% (17/64). 20% 硬 kill @ ep30 触发. β 不是根因 = structural L0 capacity 不足 + encoder 输出 lock. A2 (curriculum) 因依赖 A1 也弃. A3 (freeze encoder) 候选保留但 ROI 显著降低.
