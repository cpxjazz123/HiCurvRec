---
type: index
title: Verdicts Vault 使用说明
created: 2026-08-05
tags: [vault-entry, moc]
up: "[[index]]"
---

# Verdicts — Obsidian Vault

> **HG-Rec 复现 + κ-Stereographic 变体实验** 的所有裁定/产物（228 个文件，67 个 `.md` + 148 个 `.json` + 13 个 `.log`）。
>
> 此目录直接作为 Obsidian vault 根 — 已加 YAML frontmatter + `[[]]` 双链。

## 打开方式

```bash
# macOS
open -a "Obsidian" /home/wlia0047/ar57/wenyu/GeneRec/verdicts

# Linux (obsidian-cli)
obsidian /home/wlia0047/ar57/wenyu/GeneRec/verdicts

# 手动
# Obsidian → Open Vault → Open folder as Vault → 选 verdicts/
```

## 入口

打开后从 `index.md` (Map of Content) 入手 — 分 A/B/C/... 主题涵盖全部任务。

## 链接语法

- `[[index]]` — MOC 入口
- `[[task472_issue179_direction_a_gate4_200ep_result]]` — 具体 verdict 文件
- `issue41` (无方括号) — issue 抽象名（vault 内无同名 .md/.json，请改用具体 verdict 文件名）
- Frontmatter tag: `#hyp` / `#taskA` / `#kappa` / `#beam20-reeval` ...

## Frontmatter 模式

每个 verdict `.md` 都有：

```yaml
---
type: verdict | precheck | canary | diagnostic | analysis | status | index
issue: 41
task: 448
gate: 4
status: PASS | FAIL | PARTIAL | NO-GO | NOPE
tags: [taskA, hyp, kappa, ...]
created: 2026-08-02
up: "[[index]]"
---
```

## Dataview 查询示例（需装 Dataview plugin）

```dataview
TABLE status, type, issue, task, created
FROM "verdicts"
WHERE type = "verdict"
SORT created DESC
```

```dataview
LIST
FROM "verdicts"
WHERE status = "FAIL" AND contains(tags, "hyp")
```

## 自动化脚本（备份位于 `$CLAUDE_JOB_DIR/tmp/verdicts_backup_2026-08-05/`）

| 脚本 | 作用 |
|---|---|
| `add_obsidian_frontmatter.py` | 批量添加 YAML frontmatter |
| `patch_task_ids.py` | 补齐缺失的 `task:` 字段 |
| `obsidian_schema.md` | 整理 schema 设计文档 |

## 不属于 vault 的内容（已排除）

- `__pycache__/` —— Python 缓存，已在整理时删除

## `.json` 与 `.log` 文件

Obsidian 不展示这些，但保留供 grep / 程序访问：
- `.json`：机器 verdict / 评测指标 (148 个, e.g. `issue30_taskA_v5b_beam20_reeval.json`)
- `.log`：原始评测输出 (13 个, e.g. `taskA_v7c_fulltest_beam20.log`)

---

**整理时间**: 2026-08-05  
**整理人**: Claude (M3.1-mini)  
**基线**: Task #84 test R@10=0.1024  
**历史最佳**: hyp v2 端到端 (Task #448+ 系列) test R@10=0.1048  
**决策阈值**: test R@10 > 0.1024 → GO

> 本 vault 含两个最近 commit 的关键 verdict:
> - `taskA_hyp_v7_issue43_distance_bucket_neg_verdict.json` (Issue #43 FAIL 0.0968)
> - `task472_issue179_direction_a_gate4_200ep_result.md` (Issue #179/Task #472 NO-GO R23 触发)
