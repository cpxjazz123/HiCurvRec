---
type: experiment
category: "experiment"
zh: "方向B 建立 HG-Rec 端到端基线"
direction: "方向B"
status: "BASELINE"
topic: "baseline"
created: 2026-08-05
tags:
  - status-baseline
  - topic-baseline
  - experiment
---

# 方向B 建立 HG-Rec 端到端基线

> **情况**: [[HG-Rec 端到端基线方法|HG-Rec 端到端基线方法]] (BASELINE)
> **主题**: [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]] (baseline)

## 用什么方法

方向B HG-Rec 原论文 4 阶段流水线 (sentence-T5 + Poincaré RQ-VAE + T5-mini + beam search)

## 方法描述

**方向 B 做了什么**: 把 HG-Rec 原论文 4 阶段流水线作为 ground truth baseline 复现到方向 B。

**方法描述**:
- **Stage 1 嵌入**: Sentence-T5-base (768 维) 编码 9922 个 Musical_Instruments 商品描述为欧氏向量
- **Stage 2 量化**: Poincaré RQ-VAE 三层残差量化，每层 256 码字，3 位 → 4 位去重，得到 SID
- **Stage 3 训练**: T5-mini 自回归训练，给定用户历史 SID 序列预测下一交互 SID
- **Stage 4 评测**: Beam search 解码 (beam=20)，计算 R@5/10/20 和 NDCG@5/10/20

**作为基准**: 所有变体与 baseline 对比。**test R@10 = 0.1024** 是 ground truth。

## 关键指标

- **test_r10**: 0.1024
- **valid_r10**: 0.1267
- **ndcg20**: 0.0821

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
