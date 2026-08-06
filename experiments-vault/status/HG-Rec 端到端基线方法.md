---
type: status-node
zh: "HG-Rec 端到端基线方法"
status: "BASELINE"
created: 2026-08-05
tags:
  - status-baseline
  - node
---

# HG-Rec 端到端基线方法

> **归类情况**: BASELINE

## 用什么方法

HG-Rec 原论文端到端复现

## 方法描述

**方法**: HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 原论文四阶段流水线复现。

**方法描述**:
- **Stage 1 嵌入**: Sentence-T5-base 将 9922 个 Musical_Instruments 商品描述编码为 768 维欧氏向量
- **Stage 2 量化**: Poincaré RQ-VAE 三层残差量化 (3 位 → 4 位去重)，每层 256 个码字，得到 SID (Semantic ID)
- **Stage 3 训练**: T5-mini 自回归训练，给定用户历史 SID 序列预测下一交互 SID
- **Stage 4 评测**: Beam search 解码，R@5/10/20 + NDCG@5/10/20

**作为基准**: 所有变体与 baseline 对比。**test R@10 = 0.1024** 是 ground truth，任何变体 R@10 > 0.1024 即视为突破，否则 NO-GO。

## 本情况包含 1 个实验 + 0 个 reference

### 实验
- [[方向B 建立 HG-Rec 端到端基线|方向B 建立 HG-Rec 端到端基线]]
  > 方法: 方向B HG-Rec 原论文 4 阶段流水线 (sentence-T5 + Poincaré RQ-VAE + T5-mini + beam search)

## 相关情况

- [[Hyp 路线端到端配方方法 (历史最佳)|Hyp 路线端到端配方方法 (历史最佳)]]
- [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法]]
- [[Issue 收口裁定与协议诊断方法|Issue 收口裁定与协议诊断方法]]
- [[系列内多 step 拆解方法|系列内多 step 拆解方法]]
- [[扫描 sweep 数据集收集方法|扫描 sweep 数据集收集方法]]
