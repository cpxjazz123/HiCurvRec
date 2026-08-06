---
type: topic-node
zh: "基线复现方法 (HG-Rec 原论文流水线)"
topic: "baseline"
created: 2026-08-05
tags:
  - topic-baseline
  - node
---

# 基线复现方法 (HG-Rec 原论文流水线)

> **主题分类**: baseline

## 用什么方法

HG-Rec 原论文端到端流水线

## 方法描述

HG-Rec 4 阶段流水线: Sentence-T5-base 嵌入 → Poincaré RQ-VAE 量化 → T5-mini 训练 → Beam search 评测。**作用**: 提供 ground truth baseline，所有变体与之对比。

## 本主题下的实验

### [[方向B 建立 HG-Rec 端到端基线|方向B 建立 HG-Rec 端到端基线 (BASELINE)]]

> **方法**: 方向B HG-Rec 原论文 4 阶段流水线 (sentence-T5 + Poincaré RQ-VAE + T5-mini + beam search)


## 相关主题

- [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]]
- [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]]
- [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]]
- [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]]
- [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]
