---
type: topic-node
zh: "方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)"
topic: "early_issues"
created: 2026-08-05
tags:
  - topic-early-issues
  - node
---

# 方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)

> **主题分类**: early_issues

## 用什么方法

Stage 2/3 防线建设 + κ codebook sync

## 方法描述

对 Stage 2/3 的早期 issue 逐一裁定 + 量化器塌缩根因定位:
- 早期 issue 逐个裁定
- 量化器塌缩根因 = 欧氏 MSE 重建 + poincare commit 梯度爆炸
- 修复方案 = poincare recon + κ clamp -0.5 + κ LR 3x + mix LR 1x
- 修复后 unique_3digit 达 81%/80%

## 本主题下的实验

### [[方向B 逐个裁定早期 issue 并定位塌缩根因|方向B 逐个裁定早期 issue 并定位塌缩根因 (MIXED)]]

> **方法**: 方向B 早期 issue 裁定 + κ codebook sync + 量化器塌缩根因定位


## 相关主题

- [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]]
- [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]]
- [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]]
- [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]]
- [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]
