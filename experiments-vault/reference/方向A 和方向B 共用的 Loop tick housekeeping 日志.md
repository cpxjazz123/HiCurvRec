---
type: reference
category: "reference"
zh: "方向A 和方向B 共用的 Loop tick housekeeping 日志"
direction: "通用"
status: "DATA"
topic: "loop_tick"
created: 2026-08-05
tags:
  - status-data
  - topic-loop-tick
  - reference
---

# 方向A 和方向B 共用的 Loop tick housekeeping 日志

> **情况**: [[扫描 sweep 数据集收集方法|扫描 sweep 数据集收集方法]] (DATA)
> **主题**: [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]] (loop_tick)
> **类别**: reference (诊断/扫描/housekeeping，非主实验)

## 用什么方法

R10 v2 + R16 自动 idle 维护

## 方法描述

**这个 reference 做了什么**: 当 0 open issue + §16 backlog 空 + 无 owner 派工，自动进入 housekeeping 模式。

**方法描述**:
- 检查 GPU 全空闲 (期望 util=0%, mem=0MiB)
- 检查 in_progress 任务是否有活跃 PID + file mtime 更新 (R24)
- 清 stale `_TRAINING_PID`
- 维护 git 工作树 (housekeeping commit)
- 4 卡 GPU 全部释放

**触发器**: R10 v2 (idle) + R16 (loop 定时 tick) — 但 OPEN issue 出现时 R10 自动失效，转 R19 激进 owner 模式。

## 关键指标

- **n_idle_status_logs**: 3

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
