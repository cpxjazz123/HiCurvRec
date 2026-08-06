---
type: experiment
category: "experiment"
zh: "方向B 用 T5 容量 sweep (mini/small/base) 找甜点位"
direction: "方向B"
status: "MIXED"
topic: "t5_capacity"
created: 2026-08-05
tags:
  - status-mixed
  - topic-t5-capacity
  - experiment
---

# 方向B 用 T5 容量 sweep (mini/small/base) 找甜点位

> **情况**: [[系列内多 step 拆解方法|系列内多 step 拆解方法]] (MIXED)
> **主题**: [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]] (t5_capacity)

## 用什么方法

方向B 扫 T5 模型大小 (mini/small/base) + encoder 变体

## 方法描述

**方向 B 做了什么**: 对 T5 模型容量点做 sweep，验证 mini 是否是 HG-Rec 框架甜点位。

**方法描述**:
- **T5-mini 12M**: HG-Rec 原默认, R@10 = 0.1020
- **T5-small 60M**: HG-Rec paper 用, R@10 = 0.1020
- **T5-base 220M**: capacity unlock 尝试, PASS 但 R@10 不变
- **encoder 变体**: orc-embedding 解锁 / sigmoid/conformal 位置编码 (terminated) / 各种 stage3 训练 trick

**共同结论**: T5-mini 是 HG-Rec 框架甜点位，进一步加大容量边际收益小。

## 关键指标

- **t5_base_pass**: True
- **t5_mini_r10**: 0.102
- **t5_small_r10**: 0.102

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
