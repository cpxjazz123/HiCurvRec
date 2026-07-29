# Task #326 — K=384 sweet spot probe (task279 K-sweep expansion)

**日期**: 2026-07-30
**状态**: 📋 READY (R11.5 自主决策启动, 取代重复的 task326, 等 task320 完成后启动)
**Stage**: Stage 1 + Stage 2 + Stage 3 + Stage 4 (单臂 K0=384 探针)
**Anchor**: task194 K=256 R@10=0.1053 (anchor ⭐⭐⭐)

## 背景

task194 K-sweep K0 ∈ {32, 64, 128, 256} 实证 R@10 = {0.1034, 0.1041, 0.1027, 0.1053}. K=256 是 trade-off 顶峰.
task279 K-sweep K0 ∈ {512, 1024} 实证 R@10 = {0.0824, 0.0847} — K ≥ 512 区间 R@10 退化.
**K=384 是 K=256 → K=512 插值点, 未探索**. 

Hypothesis: K=384 是否是更优 sweet spot, 或者沿用 K ≥ 512 退化曲线?
- 若 K=384 R@10 > 0.1053 → GO (新 anchor ⭐⭐⭐⭐)
- 若 0.1020 < K=384 R@10 < 0.1053 → 数据点补完 K-sweep 曲线
- 若 K=384 R@10 ≤ 0.1020 → 退化曲线在 K ≥ 256 起始, 锁死 K=256 为 anchor

## 设计 (单臂 K0=384 probe)

| 参数 | 值 | 备注 |
|------|----|------|
| K0 (L0 capacity) | 384 | 探针 |
| L1 capacity | 128 | 跟 task194 一致 |
| L2 capacity | 256 | 跟 task194 一致 |
| codebook_size | [384, 128, 256, 1] | |
| num_epochs (Stage 1) | 200 | 跟 task194 一致 |
| num_epochs (Stage 3) | 200 | 跟 task194 一致 |
| 其他超参 | baseline | 完全复用 task194 |

## 决策阈值

| K=384 R@10 | 解读 |
|------------|------|
| > 0.1053 | GO ⭐⭐⭐⭐ (新 anchor, 跨 K-sweep 7-arm 最高) |
| 0.1020 ~ 0.1053 | 中性, 跟 K=256 anchor 持平 (数据点) |
| ≤ 0.1020 | 退化曲线起始, K=256 锁死 anchor |

## 关联

- task194 (K-sweep 4-arm {32, 64, 128, 256} → K=256 anchor ⭐)
- task279 (K-sweep K=512/1024 NO-GO, K=256 是 trade-off 顶峰)
- task326 (REPLACED — 重复 task304 D6 ablation, HALT)
- task304 (D6 ablation 闭环: r_l+s_l synergy CONFIRMED)
- loop.md §16 R10 backlog

## R10 推进

- 等 task320 完成 (200 epoch Stage 3 + Stage 4 eval)
- GPU 释放后立即启动 K=384 Stage 1 (跟 task194 一样 ~50 min)
- Stage 2 Sinkhorn → Stage 3 T5-mini 200 epoch → Stage 4 R@10 eval
- 总耗时 ~1.5-2h, 决策阈值 R@10 > 0.1053

## 关键决策点 (R11.5 自主决策)

1. **启动 K=384 probe**: 取代 task326 重复 task304 的浪费, R11.5 兜底选真正未探索方向
2. **单臂 probe (vs K=384/512 sweep)**: 只测 K=384 一个点, 假设 K=512 已 NO-GO (task279), 插值验证即可
3. **GPU 1 启动**: task320 释放后错峰 (R7 ✅ 不抢卡)

result: Task #326 — K=384 sweet spot probe. 取代重复 task326, 等 task320 完成后启动单臂 K=384, 决策阈值 R@10 > 0.1053 (vs task194 K=256 anchor)