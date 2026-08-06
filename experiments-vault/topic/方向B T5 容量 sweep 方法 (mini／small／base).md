---
type: topic-node
zh: "方向B T5 容量 sweep 方法 (mini/small/base)"
topic: "t5_capacity"
created: 2026-08-05
tags:
  - topic-t5-capacity
  - node
---

# 方向B T5 容量 sweep 方法 (mini/small/base)

> **主题分类**: t5_capacity

## 用什么方法

T5 模型大小点扫描 (mini/small/base)

## 方法描述

对 T5 模型容量点做 sweep:
- T5-mini 12M (HG-Rec 原默认)
- T5-small 60M (HG-Rec paper 用)
- T5-base 220M (capacity unlock 尝试)
- **结论**: T5-mini 是 HG-Rec 框架甜点位，进一步加大容量边际收益小

## 本主题下的实验

### [[方向B 用 T5 容量 sweep (mini／small／base) 找甜点位|方向B 用 T5 容量 sweep (mini/small/base) 找甜点位 (MIXED)]]

> **方法**: 方向B 扫 T5 模型大小 (mini/small/base) + encoder 变体


## 相关主题

- [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]]
- [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]]
- [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]]
- [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]]
- [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]
