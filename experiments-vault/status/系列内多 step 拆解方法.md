---
type: status-node
zh: "系列内多 step 拆解方法"
status: "MIXED"
created: 2026-08-05
tags:
  - status-mixed
  - node
---

# 系列内多 step 拆解方法

> **归类情况**: MIXED

## 用什么方法

系列内多 step 拆解 + 阶段性 Gate 评估

## 方法描述

**方法**: 将一个较大实验拆解为多个 step (Gate1/2/3/4) 或 issue 序列，每个 step 独立判定 PASS/FAIL，整体 Status 取 MIXED。

**方法描述**:
- **Step 1**: κ-sync 重新校准 — PASS
- **Step 2**: weighted-mixed curvature — PASS
- **Step 3**: stage2 κ-vq-loss forward path fix — 系列本身为 PASS 但与下游 T5 集成失败
- **Step 4**: stage3 training — Phase 1 部分 PASS
- **Step 5**: 方向A/B Gate4 200ep — **R@K = 0 (R23 触发)** ← 系列 NO-GO 收口

**核心洞察**: Phase1 (Gate1+2) 全部 PASS，但 Gate3 训练 val_sim 与 Gate4 真实 R@K 出现 protocol split —
**训练仿真 ≠ 真实 R@K**。argmax 解码要求 4 个 digit token 全部正确，wrapper 残差干扰 logits 偏离训练分布，导致 Gate4 全军覆没。MIXED 不是简单部分过部分败，而是 — **前 Gate PASS 后 Gate FAIL 是结构性现象**。

## 本情况包含 3 个实验 + 1 个 reference

### 实验
- [[方向A 用 T5 stage3 wrapper 注入 κ|方向A 用 T5 stage3 wrapper 注入 κ]]
  > 方法: 方向A 在 T5 stage3 加 wrapper (BoundedKappaScaleConditioner) 注入 κ
- [[方向B 用 T5 容量 sweep (mini／small／base) 找甜点位|方向B 用 T5 容量 sweep (mini/small/base) 找甜点位]]
  > 方法: 方向B 扫 T5 模型大小 (mini/small/base) + encoder 变体
- [[方向B 逐个裁定早期 issue 并定位塌缩根因|方向B 逐个裁定早期 issue 并定位塌缩根因]]
  > 方法: 方向B 早期 issue 裁定 + κ codebook sync + 量化器塌缩根因定位

### Reference (诊断/扫描/housekeeping)
- [[方向A 和方向B 共用的 19 份诊断报告|方向A 和方向B 共用的 19 份诊断报告]]
  > 方法: 19 份分主题审计报告 (代码 + 数据 + 几何 + 协议)

## 相关情况

- [[HG-Rec 端到端基线方法|HG-Rec 端到端基线方法]]
- [[Hyp 路线端到端配方方法 (历史最佳)|Hyp 路线端到端配方方法 (历史最佳)]]
- [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法]]
- [[Issue 收口裁定与协议诊断方法|Issue 收口裁定与协议诊断方法]]
- [[扫描 sweep 数据集收集方法|扫描 sweep 数据集收集方法]]
