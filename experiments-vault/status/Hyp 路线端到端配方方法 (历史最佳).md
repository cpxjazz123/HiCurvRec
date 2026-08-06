---
type: status-node
zh: "Hyp 路线端到端配方方法 (历史最佳)"
status: "BREAKTHROUGH"
created: 2026-08-05
tags:
  - status-breakthrough
  - node
---

# Hyp 路线端到端配方方法 (历史最佳)

> **归类情况**: BREAKTHROUGH

## 用什么方法

Hyp 路线端到端配方 + 自由 κ

## 方法描述

**方法**: 在 HG-Rec 框架内引入 per-item radius (Poincaré 球内自适应半径)，配合 stage 2 capmatch 层权重 +REL_STRUCT 关系结构先验，让 κ 自由学习，跳出固定 c=1 的几何。

**方法描述**:
- **Stage 1**: per-item radius 设计 — `r = R_MAX · sigmoid(3·(||t5_emb|| − 0.7))`，R_MAX=0.99，让每个 item 自适应选择球内位置 (浅层 r 更大，深层 r 更小)
- **Stage 2**: capmatch 配方 — `REC_LAYER_W = 1:3:9` (浅层权重 1，中层 3，深层 9，损失反向促进深层码本充分训练) + `REL_STRUCT=1` (加入关系结构损失) + bf16 + bs1024 + 1000 epoch
- **Stage 3**: T5-mini bf16 + bs1024 + lr=3e-4 + 95 epoch 早停 (early_stop=20)

**结果**: test R@10 = **0.1048** (+0.0024 vs baseline)，valid R@10 = 0.1269 (+0.0002)，NDCG@20 = 0.0860 (+0.0105)。**仅此一例** 稳定超基线，5% 超基线 (0.1075) 在 hyp 路线物理不可达已 9 次确认。

## 本情况包含 1 个实验 + 0 个 reference

### 实验
- [[方向A 用 Hyp 端到端配方达成历史最佳|方向A 用 Hyp 端到端配方达成历史最佳]]
  > 方法: 方向A per-item radius + capmatch REL_STRUCT + 自由 κ (Hyp 路线唯一超基线)

## 相关情况

- [[HG-Rec 端到端基线方法|HG-Rec 端到端基线方法]]
- [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法]]
- [[Issue 收口裁定与协议诊断方法|Issue 收口裁定与协议诊断方法]]
- [[系列内多 step 拆解方法|系列内多 step 拆解方法]]
- [[扫描 sweep 数据集收集方法|扫描 sweep 数据集收集方法]]
