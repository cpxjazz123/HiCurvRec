---
type: topic-node
zh: "Hyp 路线方法 (per-item radius + 自由 κ)"
topic: "hyp_routes"
created: 2026-08-05
tags:
  - topic-hyp-routes
  - node
---

# Hyp 路线方法 (per-item radius + 自由 κ)

> **主题分类**: hyp_routes

## 用什么方法

per-item radius + 自由 κ (Poincaré 球内嵌入)

## 方法描述

在 HG-Rec 框架内引入**几何 (双曲空间) 增强**: 
- 每个商品在 Poincaré 球内有独立半径 (r = R_MAX·sigmoid(...))，浅层半径更大、深层更小
- κ (曲率参数) 在 3 层码本上**自由学习**，不锁 c=1
- 配合 capmatch 层权重 (浅层 1，中层 3，深层 9) + REL_STRUCT 关系结构损失
- **结果**: 方向A 用 Hyp 路线达成历史最佳 (test R@10 = 0.1048)，但 5% 超基线 (0.1075) 不可达

## 本主题下的实验

### [[方向A 用 Hyp 端到端配方达成历史最佳|方向A 用 Hyp 端到端配方达成历史最佳 (BREAKTHROUGH)]]

> **方法**: 方向A per-item radius + capmatch REL_STRUCT + 自由 κ (Hyp 路线唯一超基线)

### [[方向A 试图用 trust region + anchored 稳定 κ|方向A 试图用 trust region + anchored 稳定 κ (NO_GO)]]

> **方法**: 方向A 用 trust region + anchored κ 稳定化 (NO-GO)

### [[方向A 用距离分桶替换 KMeans 负采样|方向A 用距离分桶替换 KMeans 负采样 (NO_GO)]]

> **方法**: 方向A 用 cosine 距离分位区间负采样 + false-neg 排除 (NO-GO)


## 相关主题

- [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]]
- [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]]
- [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]]
- [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]]
- [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]
