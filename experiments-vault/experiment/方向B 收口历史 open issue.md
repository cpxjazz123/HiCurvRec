---
type: experiment
category: "experiment"
zh: "方向B 收口历史 open issue"
direction: "方向B"
status: "FAIL"
topic: "issue_closure"
created: 2026-08-05
tags:
  - status-fail
  - topic-issue-closure
  - experiment
---

# 方向B 收口历史 open issue

> **情况**: [[Issue 收口裁定与协议诊断方法|Issue 收口裁定与协议诊断方法]] (FAIL)
> **主题**: [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]] (issue_closure)

## 用什么方法

方向B 4 Gate 协议裁定 + 历史 issue 最终收口

## 方法描述

**方向 B 做了什么**: 对历史 4 个 open issue 做最终收口裁定，含 1 个极端 FAIL 案例。

**方法描述**:
- **Issue #40**: source 切换 (data source change → R@10 reset) — 系统性重置
- **Issue #55**: mode collapse (量化器死亡码本) — 修复 + NO-GO
- **Issue #57**: Gate0 Stage4 broken baseline (system issue) — NO-GO
- **Issue #76**: curvprior 端到端 **test R@10 = 0.047** — 极端 FAIL

**核心教训 (Issue #76)**:
- 新 SID 与 T5 不兼容，sanity=0 证明 T5 完全不认新码字
- κ 本身非问题：标准 SID+κ 对照 R@10 = 0.11 (但 #76 错在 SID 不在 κ)
- curvprior 路线**整体收线**

## 关键指标

- **issue76_test_r10**: 0.047
- **issue76_sanity_check**: 0

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
