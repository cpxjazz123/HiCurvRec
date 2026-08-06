---
type: reference
category: "reference"
zh: "方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集"
direction: "通用"
status: "DATA"
topic: "issue30_beam20"
created: 2026-08-05
tags:
  - status-data
  - topic-issue30-beam20
  - reference
---

# 方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集

> **情况**: [[扫描 sweep 数据集收集方法|扫描 sweep 数据集收集方法]] (DATA)
> **主题**: [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]] (issue30_beam20)
> **类别**: reference (诊断/扫描/housekeeping，非主实验)

## 用什么方法

α/β 扫描 + beam width 扫描 (大文件集, 60+ json)

## 方法描述

**这个 reference 做了什么**: 系统扫描 α/β/beam 三维度，作为后续分析的基础材料。

**方法描述**:
- **α sweep**: 15 个点 (logit -5.3 到 -1.0)
- **β sweep**: 5 个点 (2.4 到 4.6)
- **beam width**: 6 个宽度 (beam=1/5/10/20/50/80/100)
- **变体覆盖**: 方向A hyp 多版本 + 方向B v5/v5a1/v6 = 25 变体 × 60+ json 文件

**核心结论 (数据集层面的 verdict)**:
- beam20 不改变胜负格局，R@10 排序与 beam=1 一致
- baseline ceiling 由 SID+T5 共同决定，decode 宽度不是瓶颈

## 关键指标

- **n_json_files**: 60+
- **variants**: 25
- **beam_widths**: 6

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
