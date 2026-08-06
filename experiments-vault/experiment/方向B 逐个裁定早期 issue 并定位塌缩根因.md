---
type: experiment
category: "experiment"
zh: "方向B 逐个裁定早期 issue 并定位塌缩根因"
direction: "方向B"
status: "MIXED"
topic: "early_issues"
created: 2026-08-05
tags:
  - status-mixed
  - topic-early-issues
  - experiment
---

# 方向B 逐个裁定早期 issue 并定位塌缩根因

> **情况**: [[系列内多 step 拆解方法|系列内多 step 拆解方法]] (MIXED)
> **主题**: [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]] (early_issues)

## 用什么方法

方向B 早期 issue 裁定 + κ codebook sync + 量化器塌缩根因定位

## 方法描述

**方向 B 做了什么**: 对 Stage 2/3 早期 issue 逐个裁定，并定位量化器塌缩根因。

**方法描述**:
- 早期 issue 逐个裁定 (κ-scale 诊断 / mixed-weight 诊断 / baseline 端到端 4 Step / κ codebook sync / long run canary 收口)
- **量化器塌缩根因 = 欧氏 MSE 重建 + poincare commit 梯度爆炸**
- **修复方案 = poincare recon + κ clamp -0.5 + κ LR 3x + mix LR 1x**
- 修复后 unique_3digit 达 81%/80% (taskA/taskB)
- 双臂 option A/B NO-GO 收线
- long run 75+ epoch 后回归 canary 收口

**结果**: 防线建设完成，但尝试的所有 option A/B 配方均 NO-GO。

## 关键指标

- **unique_3digit**: 81%/80%
- **stage2_fix**: PASS

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
