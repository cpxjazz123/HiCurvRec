# Task #47 — 战线二 L1 (d) GSRQ Gain-Shape

> **任务目的**: 验证 Gain-Shape RQ (gain 标量量化 + shape 球面 RQ) 是否能在 norm 长尾上工作
> **完成日期**: 2026-07-20
> **状态**: ⛔ 已合并到 Task #46

---

## 1. 背景

承接 Task #46 L1 (a)(b)(c) 3 种 norm 修复. GSRQ 把 norm 单独量化作为 gain, shape 走球面 RQ — 是另一种 norm 修复路线.

## 2. 决策

log1p 在 Task #46 中 SCR = 0.31x (远低于 GSRQ 历史数据), GSRQ 路线收益不显著, **合并/搁置**. 主线用 log1p.

## 3. 产物

无新文件 (Task #46 已覆盖 norm 修复).
