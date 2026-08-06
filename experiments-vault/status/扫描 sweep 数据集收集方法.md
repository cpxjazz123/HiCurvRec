---
type: status-node
zh: "扫描 sweep 数据集收集方法"
status: "DATA"
created: 2026-08-05
tags:
  - status-data
  - node
---

# 扫描 sweep 数据集收集方法

> **归类情况**: DATA

## 用什么方法

扫描 / sweep 数据集收集

## 方法描述

**方法**: 系统扫描 / 大规模参数 sweep 后收集的数据集，**不形成 GO/NO-GO 决策**，只作为后续分析 / 多维比较的基础材料。

**方法描述**:
- **Scope 1**: Issue #30 α-sweep + beam20 reeval 数据集 (60+ json)
  - 涵盖方向A v5/v6/v7/v8 + 方向B v5/v5a1/v6 共 25 变体
  - α 参数从 logit -5.3 到 -1.0 共 15 个点扫描
  - β 参数从 2.4 到 4.6 共 5 个点扫描
  - beam width 从 1 到 100 共 6 个宽度
  - **核心结论**: beam20 不改变胜负格局，R@10 排序与 beam=1 一致
- **Scope 2**: Loop tick housekeeping log (R10 v2 idle 状态)

**用法**: 当其他实验需要做 baseline 对比 / 多点分析时，**引用此处数据集**。

## 本情况包含 0 个实验 + 2 个 reference

### Reference (诊断/扫描/housekeeping)
- [[方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集|方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集]]
  > 方法: α/β 扫描 + beam width 扫描 (大文件集, 60+ json)
- [[方向A 和方向B 共用的 Loop tick housekeeping 日志|方向A 和方向B 共用的 Loop tick housekeeping 日志]]
  > 方法: R10 v2 + R16 自动 idle 维护

## 相关情况

- [[HG-Rec 端到端基线方法|HG-Rec 端到端基线方法]]
- [[Hyp 路线端到端配方方法 (历史最佳)|Hyp 路线端到端配方方法 (历史最佳)]]
- [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法]]
- [[Issue 收口裁定与协议诊断方法|Issue 收口裁定与协议诊断方法]]
- [[系列内多 step 拆解方法|系列内多 step 拆解方法]]
