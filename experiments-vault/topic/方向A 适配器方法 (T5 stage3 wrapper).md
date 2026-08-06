---
type: topic-node
zh: "方向A 适配器方法 (T5 stage3 wrapper)"
topic: "direction_c"
created: 2026-08-05
tags:
  - topic-direction-c
  - node
---

# 方向A 适配器方法 (T5 stage3 wrapper)

> **主题分类**: direction_c

## 用什么方法

T5 stage3 加 wrapper (BoundedKappaScaleConditioner)

## 方法描述

不改 stage2，把 **κ 学习编码到 T5 stage3 旁路**:
- 在 T5 encoder/decoder 之间插入 adapter wrapper (BoundedKappaScaleConditioner)
- α 健康增长 (5.00e-1 bound at ep3)
- **失败原因**: Stage4 真实 R@K = 0 (R23 触发)。**protocol split**: 训练 loss 下降 ≠ generate/R@K 提升

## 本主题下的实验

### [[方向A 用 T5 stage3 wrapper 注入 κ|方向A 用 T5 stage3 wrapper 注入 κ (MIXED)]]

> **方法**: 方向A 在 T5 stage3 加 wrapper (BoundedKappaScaleConditioner) 注入 κ


## 相关主题

- [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]]
- [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]]
- [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]]
- [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]]
- [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]
