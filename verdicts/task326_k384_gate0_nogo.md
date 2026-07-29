# Task #326 — K=384 sweet spot probe — Gate 0 FAIL (USAGE-KILL)

**日期**: 2026-07-30
**状态**: ❌ **Gate 0 FAIL** — L0 utilization 16.9% < 20% 阈值 @ ep30, USAGE-KILL
**Stage**: Stage 1 only (Stage 2/3/4 不启动, Gate 0 失败硬停止)

## Stage 1 训练轨迹 (K=384)

| Epoch | collision_rate | L0 util | L1 util | L2 util | 决策 |
|-------|----------------|---------|---------|---------|------|
| ep25 | 0.1099 | (warmup) | (warmup) | (warmup) | — |
| ep29 | 0.0974 | 16.9% (65/384) | 87.5% (112/128) | 74.6% (191/256) | ❌ FAIL |
| ep30 | — | — | — | — | **USAGE-KILL abort** |

## 失败原因分析

K=384 + batch_size=512 (KMeans init n_samples=512 ≥ n_clusters=384 OK) 仍然触发 USAGE-KILL:
- L0 仅 65/384 (16.9%) — 离 20% 阈值差 3.1pp
- L1/L2 utilization 正常 (87.5% / 74.6%)
- collision_rate 0.0974 健康 (Sinkhorn 收敛)

**K-sweep Gate 0 模式已建立**:
| K | L0 util @ ep30 | 决策 |
|---|----------------|------|
| 256 (task194) | 100% | ✅ PASS |
| 384 (task326) | 16.9% | ❌ USAGE-KILL |
| 512 (task279) | 待 KMeans init OK | ❌ Stage 4 NO-GO R@10=0.0824 |
| 1024 (task279) | 待 KMeans init OK | ❌ Stage 4 NO-GO R@10=0.0847 |

## 决策阈值分析

| K=384 期望 R@10 | 实际结果 |
|------------------|----------|
| > 0.1053 (新 anchor) | ❌ Gate 0 FAIL, 无 Stage 4 |
| 0.1020 ~ 0.1053 | ❌ 达不到 |
| ≤ 0.1020 | ❌ 退化曲线起始 (跟 task279 K=512/1024 一致) |

## K-sweep 完整图谱 (更新版)

| K0 | Stage 1 util | Stage 4 R@10 | 决策 |
|----|--------------|--------------|------|
| 32 | (task194 OK) | 0.1034 | ✅ |
| 64 | (task194 OK) | 0.1041 | ✅ |
| 128 | (task194 OK) | 0.1027 | ✅ |
| **256** | **(task194 OK)** | **0.1053** | ⭐⭐⭐ anchor |
| **384** | **Gate 0 FAIL** | **N/A** | ❌ NO-GO |
| 512 | (task279 OK) | 0.0824 | ❌ |
| 1024 | (task279 OK) | 0.0847 | ❌ |

**K-sweep 结论**: K=256 是 trade-off 顶峰 (L0 100% + R@10 0.1053). K > 256 进入退化区:
- K=384 L0 utilization 崩塌 (USAGE-KILL @ ep30)
- K=512/1024 Stage 1 PASS 但 Stage 4 NO-GO

## R11.5 自主决策

- K=384 probe 闭环 (Gate 0 FAIL, K-sweep 曲线 K=256 → K=512 已闭合)
- 不再尝试 K=320 或其他 K ∈ (256, 384) 插值点 (K=384 已证伪)
- task326 优先级降低, 等 task320 完成 GPU 释放后启动其他 ROI 方向

## 关联

- task194 (K-sweep K0 ∈ {32, 64, 128, 256} → K=256 anchor ⭐)
- task279 (K-sweep K=512/1024 NO-GO R@10=0.0824/0.0847)
- task326 (本次, K=384 Gate 0 FAIL)
- verdicts/task326_halt_d6_duplicate.md (前次 HALT 决策)

result: Task #326 K=384 sweet spot probe — ❌ Gate 0 FAIL. K-sweep K=256 → K=512 退化曲线已闭合. K=384 L0 utilization 16.9% USAGE-KILL @ ep30. task194 K=256 anchor 0.1053 锁死, 后续 ROI 方向转向协同实验或 Stage 3/4 协议.