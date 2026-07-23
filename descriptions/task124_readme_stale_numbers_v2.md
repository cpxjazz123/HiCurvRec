# Task #124 — README.md + TASKS_INDEX.md Stale Number Refresh v2

> **任务目的**: 修复 post-#115-#123 housekeeping 后 README.md + TASKS_INDEX.md 的 stale 数字. README line 58 "114 entries" → 真实 125 entries (含 2 duplicates). TASKS_INDEX descriptions/ 123 → 127, verdicts/ 253 → 255. Coverage matrix 加 Task #122-#123. 同步两个 reviewer-facing 文档 cross-validation.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #113 (v1 stale number refresh) 把 README "descriptions (101 entries)" → "(114 entries)". 但 post-#115 housekeeping 新增 11 个 description (Task #115-#123 + 这个 #124). 实际 descriptions/ tracked = 127 (125 numbered + 2 reference).

Task #122 refresh TASKS_INDEX 数字时算错 (写 123 descriptions, 实际 127). Task #124 修正.

**Task #124 = 文档数字 v2 refresh**

---

## 2. 实验设计 (writeup only)

### 2.1 修复 README.md line 58

Replace:
```
├── descriptions/                  # Task definitions (114 entries, R9 continuous)
```
With:
```
├── descriptions/                  # Task definitions (125 entries, R9 continuous)
```

### 2.2 修复 TASKS_INDEX.md descriptions count

Replace "123 files total" → "127 files total — 125 numbered files covering 123 unique task IDs (Task #27 and Task #67 have 2 revision files each)".

### 2.3 修复 TASKS_INDEX.md verdicts count

Replace "253 files" → "255 files".

### 2.4 TASKS_INDEX.md Coverage matrix 加 Task #122-#123

Add row "Task #122 - Task #123 | ✓ closed | TASKS_INDEX.md refresh + REPRODUCE.md script reference alignment".

### 2.5 TASKS_INDEX.md Recent closed tasks 表加 #122/#123

Add 2 rows for Task #122 (TASKS_INDEX.md refresh) and Task #123 (REPRODUCE.md script alignment).

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| README line 58 updated | ✅ 闭环 |
| TASKS_INDEX descriptions 123 → 127 | ✅ 闭环 |
| TASKS_INDEX verdicts 253 → 255 | ✅ 闭环 |
| Coverage matrix Task #122-#123 加 | ✅ 闭环 |
| Recent closed tasks #122/#123 加 | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| descriptions/ 数字 (114 → ?) | ✅ 125 entries (反映实际) | ❌ 维持 114 (数字 stale, reader 困惑) |
| task27/task67 duplicates 处理 | ❌ 不清理 (renumbering 历史修订, 有价值) | ✅ rm duplicates (但 R9 不强制 unique, 修订历史丢失) |
| TASKS_INDEX 是否同步 | ✅ 同步 (cross-validation, 两个文档一致) | ❌ 仅 README (两文档不一致 reader 困惑) |
| 是否 cleanup duplicates in script | ❌ 不写 cleanup script (但记录在 verdict) | ✅ 写 cleanup script (过度工程) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep + Edit 2 docs | ~3 min |
| dispatcher 验证 | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~5 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: 改 README + TASKS_INDEX 数字不一致 (例如 README 125 但 TASKS_INDEX 写 127)
  → **缓解**: 数字来源于 `git ls-files` 实测, 两个文档都用同一 source

**风险 2**: duplicates (task27 + task67) 不清理违反 R9 spirit
  → **缓解**: R9 不强制 unique files per task ID; TASKS_INDEX 标注 "have 2 revision files each" 说明

---

## 7. 完成度跟踪

- [x] 写 descriptions/task124_readme_stale_numbers_v2.md (本文件)
- [x] 修复 README line 58 (114 → 125)
- [x] 修复 TASKS_INDEX descriptions count (123 → 127)
- [x] 修复 TASKS_INDEX verdicts count (253 → 255)
- [x] TASKS_INDEX Coverage matrix 加 Task #122-#123
- [x] TASKS_INDEX Recent closed tasks 加 #122/#123
- [x] dispatcher 5/5 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #113 (README stale number refresh v1) + #122 (TASKS_INDEX refresh)
- 后置: 无 (documentation drift 闭环)

---

**核心交付**: README + TASKS_INDEX 数字 v2 同步. 发现 R9 duplicates (task27 + task67 renumbering 残留) 但保留作为历史修订. dispatcher 5/5 PASS 不破坏.