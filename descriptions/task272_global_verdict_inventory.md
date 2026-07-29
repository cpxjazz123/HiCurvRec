# Task #272 — Issue #10/#17 全关后 VERDICT/PRODUCTS 全局 inventory

## 背景

2026-07-28 关闭 Issue #10 (collision 口径统一 + 自决 NO-GO), 2026-07-28 关闭 Issue #17 (Stage 1 utilization 强制化 + 3-Gate 全 PASS). 

至此所有 GitHub issues (#1-#17) 都是 closed 状态 (10 个 COMPLETED + 7 个 NOT_PLANNED). 项目完成一次系统性回看时机 — 把当前的项目状态、决策记录、可复现资产、未决问题整合到一份 inventory.

## 任务范围

1. 列出所有 17 个 GitHub issues 的最终状态 (COMPLETED vs NOT_PLANNED) + 关闭原因
2. 列出当前所有 verdicts 中保留 R@10 数字的 baseline + 阶段产物 (Stage 1/2/3/4 ckpt)
3. 列出 backlog 待办 (R10 + R11.5 候选 1-5)
4. 0 GPU 本 cron tick
5. 文档化到 `verdicts/task272_global_verdict_inventory_result.md` 作为项目当前态

## 关键决策点 (R11.3)

- **0 GPU inventory**: 不主动启动新训练, 仅整理
- **不假装 success**: 列出 §6.7.4 stop-loss (i) 触发历史 (task253 L0=73.44%, task271 A1 L0=26.6%), 跟 φ-paper-grade 业绩保持距离
- **不立新 issue**: Issue #5 #4 #3 #2 #1 跟当前 backlog 候选重叠, 不重提

## 物理产物

```
descriptions/task272_global_verdict_inventory.md  (本文件)
verdicts/task272_global_verdict_inventory_result.md  (主 inventory 文档)
```

result: Task #272 — 17 个 GitHub issues 状态 + 关键 verdict R@10 表 + backlog 候选 (0 GPU inventory). 项目当前态 consolidated 文档.
