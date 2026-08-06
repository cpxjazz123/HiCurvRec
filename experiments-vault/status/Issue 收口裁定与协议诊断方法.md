---
type: status-node
zh: "Issue 收口裁定与协议诊断方法"
status: "FAIL"
created: 2026-08-05
tags:
  - status-fail
  - node
---

# Issue 收口裁定与协议诊断方法

> **归类情况**: FAIL

## 用什么方法

Issue 收口裁定 + 协议诊断

## 方法描述

**方法**: 通过 4 Gate 协议 (Gate1 precheck / Gate2 quantizer / Gate3 train / Gate4 eval) 对每个变体做裁定，任一 Gate FAIL 即归为单次失败。

**方法描述**:
- **Gate 1 (precheck)**: κ_grad finite nonzero、no NaN/Inf、init_c > 0、wrapper 不破坏数据通路
- **Gate 2 (quantizer)**: util_3digit > 80%, reload_5of5_consistent, κ_per_layer_std 健康
- **Gate 3 (train)**: loss 下降无 plateau / NaN，ckpt 强制每 epoch 末存 (R12)
- **Gate 4 (eval)**: 真实 Stage4 R@K 实测 (protocol split: 训练仿真 ≠ 真实 R@K)

**FAIL 案例**:
- curvprior 端到端 **test R@10 = 0.047** — 新 SID 与 T5 不兼容，sanity=0 证明 T5 完全不认新码字
- Gate3b recontinue 终止

**主要根因**: Sid-incompat (新码字超出 T5 训练分布)，与 κ 学习本身关系小。

## 本情况包含 1 个实验 + 0 个 reference

### 实验
- [[方向B 收口历史 open issue|方向B 收口历史 open issue]]
  > 方法: 方向B 4 Gate 协议裁定 + 历史 issue 最终收口

## 相关情况

- [[HG-Rec 端到端基线方法|HG-Rec 端到端基线方法]]
- [[Hyp 路线端到端配方方法 (历史最佳)|Hyp 路线端到端配方方法 (历史最佳)]]
- [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法]]
- [[系列内多 step 拆解方法|系列内多 step 拆解方法]]
- [[扫描 sweep 数据集收集方法|扫描 sweep 数据集收集方法]]
