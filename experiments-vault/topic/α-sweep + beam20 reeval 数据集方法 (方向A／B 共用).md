---
type: topic-node
zh: "α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)"
topic: "issue30_beam20"
created: 2026-08-05
tags:
  - topic-issue30-beam20
  - node
---

# α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)

> **主题分类**: issue30_beam20

## 用什么方法

α/β 扫描 + beam width 扫描 (大文件集)

## 方法描述

**方法**: 探索 α 系数 + β 系数 + beam width 三维度全量扫描
- **α sweep**: 15 个点 (logit -5.3 到 -1.0)
- **β sweep**: 5 个点 (2.4 到 4.6)
- **beam width**: 6 个宽度 (1, 5, 10, 20, 50, 80, 100)
- **变体覆盖**: 方向A + 方向B = 25 变体
- **核心结论**: beam20 不改变胜负格局, decode 宽度不是瓶颈

## 本主题下的实验

### Reference (诊断/扫描)
- [[方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集|方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集]]
  > 方法: α/β 扫描 + beam width 扫描 (大文件集, 60+ json)


## 相关主题

- [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]]
- [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]]
- [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]]
- [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]]
- [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]
