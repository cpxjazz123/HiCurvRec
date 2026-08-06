---
type: experiment
category: "experiment"
zh: "方向A 用 T5 stage3 wrapper 注入 κ"
direction: "方向A"
status: "MIXED"
topic: "direction_c"
created: 2026-08-05
tags:
  - status-mixed
  - topic-direction-c
  - experiment
---

# 方向A 用 T5 stage3 wrapper 注入 κ

> **情况**: [[系列内多 step 拆解方法|系列内多 step 拆解方法]] (MIXED)
> **主题**: [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]] (direction_c)

## 用什么方法

方向A 在 T5 stage3 加 wrapper (BoundedKappaScaleConditioner) 注入 κ

## 方法描述

**方向 A 做了什么**: 不改 stage2，把 κ learning 编码到 T5 stage3 旁路 (BoundedKappaScaleConditioner wrapper)。

**方法描述**:
- 在 T5 encoder/decoder 之间插入 adapter wrapper
- Wrapper: BoundedKappaScaleConditioner 强制 α ≤ 0.5 bound + κ/scale 元数据

**4 Gate 流程**:
- **Gate 2 (Phase1)**: κ-sync 重新校准 + weighted-mixed curvature — **PASS**
- **Gate 3 (Phase2)**: stage3 训练 loss 健康 (9.13→1.04, -89%) 但 val_sim 与 Stage4 split
- **Gate 4 (Phase3)**: Stage4 真实 R@K = **0 (R23 触发)**

**核心教训 (protocol split)**:
- val_R@10_sim (训练仿真) = 0.115
- Stage4 R@10 (真实 argmax) = 0.0000
- **训练 loss 下降 ≠ generate/R@K 提升**
- argmax 解码要求 4 个 digit token 全部正确，wrapper 残差干扰 logits 偏离训练分布

## 关键指标

- **stage4_r10**: 0.0
- **val_sim**: 0.115
- **val_ndcg20**: 0.0972

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
