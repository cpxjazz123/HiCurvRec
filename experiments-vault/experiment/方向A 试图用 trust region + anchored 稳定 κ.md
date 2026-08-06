---
type: experiment
category: "experiment"
zh: "方向A 试图用 trust region + anchored 稳定 κ"
direction: "方向A"
status: "NO_GO"
topic: "hyp_routes"
created: 2026-08-05
tags:
  - status-no_go
  - topic-hyp-routes
  - experiment
---

# 方向A 试图用 trust region + anchored 稳定 κ

> **情况**: [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法]] (NO_GO)
> **主题**: [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]] (hyp_routes)

## 用什么方法

方向A 用 trust region + anchored κ 稳定化 (NO-GO)

## 方法描述

**方向 A 做了什么**: 在 Hyp 路线最佳配方的基础上加 κ 稳定化手段 (trust region + EMA + anchored)，试图让 κ 学得更稳，结果 test R@10 全部不超基线。

**方法描述**:
- **方法 1 (trust region + EMA)**: 限制 κ 每步更新幅度 + 动量均值平滑，限制 c 在 [1, 10] 健康区
- **方法 2 (anchored κ)**: `κ_eff = anchor + tanh(drift)·range`，anchor 选自历史最优 [0.024, 0.046, 0.056]

**两种方法的共同根因**: κ 演化路径 (而非最终值) 对 T5 训练敏感。
锚定机制虽然保留了梯度，但破坏了 Hyp 路线的 κ 训练轨迹，导致 T5 看到的隐空间分布与最佳配方不同。

**结果**: 
- anchored: test R@10 = **0.0993** (-0.0031)
- trust region: test R@10 = 0.1002 (-0.0022)

**结论**: 单纯稳定 κ 值不能保证端到端超基线，**整体收线**。

## 关键指标

- **anchored_test_r10**: 0.0993
- **delta_anchored**: -0.0031
- **tr_test_r10**: 0.1002

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
