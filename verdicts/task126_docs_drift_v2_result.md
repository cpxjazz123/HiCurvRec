# Task #126 — TASKS_INDEX + CHANGELOG Documentation Drift v2 (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: TASKS_INDEX.md + CHANGELOG.md docs drift v2 refresh. TASKS_INDEX coverage matrix 加 "Task #124 - #125" 行; Recent closed tasks 标题 "Task #101 - Task #121" → "Task #101 - Task #125", 表格加 #122/#123/#124/#125 4 行 (Task #122/#123 之前漏加, #124/#125 缺失); CHANGELOG [Unreleased] Notes 段追加 #122-#125 housekeeping 续集. dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| TASKS_INDEX coverage matrix + #124-#125 行 | `grep "Task #124 - #125" TASKS_INDEX.md` | ✅ |
| TASKS_INDEX Recent closed tasks 标题 | `grep "Task #101 - Task #125" TASKS_INDEX.md` | ✅ |
| TASKS_INDEX Recent closed tasks 加 4 行 | `grep -cE "^\| #(122|123|124|125) " TASKS_INDEX.md` | ✅ 4 |
| CHANGELOG Notes 段追加 | `grep "#122-#125" CHANGELOG.md` | ✅ |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ |

---

## 2. Before / After

### TASKS_INDEX.md coverage matrix (before)

```
| Task #122 - Task #123 | ✓ closed | TASKS_INDEX.md refresh + REPRODUCE.md script reference alignment |
```

### TASKS_INDEX.md coverage matrix (after)

```
| Task #122 - Task #123 | ✓ closed | TASKS_INDEX.md refresh + REPRODUCE.md script reference alignment |
| Task #124 - Task #125 | ✓ closed | README + TASKS_INDEX stale number refresh v2 + CI workflow dispatcher sync |
```

### TASKS_INDEX.md Recent closed tasks (before)

- Header: `## Recent closed tasks (Task #101 - Task #121)`
- Last row: `#121 | Post-submission housekeeping synthesis + CHANGELOG [Unreleased]`

### TASKS_INDEX.md Recent closed tasks (after)

- Header: `## Recent closed tasks (Task #101 - Task #125)`
- Appended rows:
  - `#122 | TASKS_INDEX.md refresh (Task #113-#121 coverage) | ✅ | verdicts/task122_tasks_index_refresh_result.md`
  - `#123 | REPRODUCE.md script reference alignment (7 nonexistent → 8 real) | ✅ | verdicts/task123_reproduce_md_script_alignment_result.md`
  - `#124 | README + TASKS_INDEX stale number refresh v2 (114→125, 253→255) | ✅ | verdicts/task124_readme_stale_numbers_v2_result.md`
  - `#125 | CI workflow sync with 5-audit dispatcher (4 separate → 1 dispatcher) | ✅ | verdicts/task125_ci_workflow_dispatcher_sync_result.md`

### CHANGELOG.md [Unreleased] Notes (before)

```
- Post-submission housekeeping cycle: Tasks #111 ... → #121 (this synthesis).
```

### CHANGELOG.md [Unreleased] Notes (after)

```
- Post-submission housekeeping cycle: Tasks #111 ... → #121 (synthesis) →
  #122-#125 (continued housekeeping: TASKS_INDEX refresh + REPRODUCE.md
  script alignment + README/TASKS_INDEX stale numbers v2 + CI workflow
  dispatcher sync).
```

---

## 3. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| TASKS_INDEX refresh 范围 | ✅ 补 #122-#125 (跨 4 个 task cross-validation) | ❌ 只补 #124/#125 (#122/#123 也漏加, 不一致) |
| CHANGELOG [Unreleased] 改不改 | ✅ Notes 段追加 (Keep-a-Changelog 推荐) | ❌ 不改 (但 [Unreleased] 应跟当前一致) |
| 是否 gitignore src/configs/tools | ❌ 不动 (REPRODUCE §0 onboarding 期望 visible) | ✅ gitignore (但破坏 onboarding verify step) |
| 是否做 v3 (更多 housekeeping) | ❌ 不做 (drift 已闭环, 边际收益 < 5 min) | ✅ 继续找 (over-engineering) |

---

## 4. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| TASKS_INDEX coverage matrix 含 #124-#125 | `grep "Task #124 - #125" TASKS_INDEX.md` | ✅ |
| TASKS_INDEX Recent closed tasks 标题更新 | `grep "Task #101 - Task #125" TASKS_INDEX.md` | ✅ |
| TASKS_INDEX 4 行新加 (#122-#125) | `grep -cE "^\| #(122|123|124|125) \|" TASKS_INDEX.md` | ✅ 4 |
| CHANGELOG Notes 段含 #122-#125 | `grep "#122-#125" CHANGELOG.md` | ✅ |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |

---

## 5. 关联

- 前置: Task #121 (post-submission housekeeping synthesis) + #122-#125 (4 post-#121 housekeeping)
- 后置: 无 (post-#121 housekeeping cycle 续集闭环, reviewer-facing docs 现在跟 git 状态一致)

---

result: Task #126 — TASKS_INDEX + CHANGELOG docs drift v2 闭环. 3 处 drift 同步到 #125 状态: TASKS_INDEX coverage matrix +1 行, Recent closed tasks 标题 + 4 行, CHANGELOG [Unreleased] Notes 段 +1 句. dispatcher 5/5 PASS 不破坏. reviewer-facing docs 现在跟 git 状态 cross-validation 闭合.
