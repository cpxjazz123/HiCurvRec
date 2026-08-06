---
type: topic-node
zh: "Loop tick housekeeping 方法 (方向A/B 共用)"
topic: "loop_tick"
created: 2026-08-05
tags:
  - topic-loop-tick
  - node
---

# Loop tick housekeeping 方法 (方向A/B 共用)

> **主题分类**: loop_tick

## 用什么方法

R10 v2 + R16 自动 idle 维护

## 方法描述

**方法**: 当 0 open issue + §16 backlog 空 + 无 owner 派工时，自动 housekeeping
- 检查 GPU 状态 (期望 util=0%, mem=0MiB)
- 检查 in_progress 任务是否有活跃 PID + file mtime 更新
- 清 stale `_TRAINING_PID`
- 维护 git 工作树 + housekeeping commit
- **触发器**: R10 v2 (idle) — 但 OPEN issue 出现时 R10 自动失效，转 R19 激进 owner 模式

## 本主题下的实验

### Reference (诊断/扫描)
- [[方向A 和方向B 共用的 Loop tick housekeeping 日志|方向A 和方向B 共用的 Loop tick housekeeping 日志]]
  > 方法: R10 v2 + R16 自动 idle 维护


## 相关主题

- [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]]
- [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]]
- [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]]
- [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]]
- [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]]
- [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]
