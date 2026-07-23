# Task #122 — TASKS_INDEX.md Refresh for Tasks #113-#121 + Post-Submission Housekeeping

> **任务目的**: TASKS_INDEX.md 之前 snapshot 在 Task #112 (v1.0.0 release artifacts), 缺少 Task #113-#121 coverage. 刷新: (a) 总 task 数 114 → 121, (b) 加 Task #113-#121 到 coverage matrix, (c) 加 Recent closed tasks 表格, (d) 加 Post-submission housekeeping artifacts 表格, (e) 更新 Conventions 段反映 R8 §9.3 + R9 enforcement.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

TASKS_INDEX.md (Task #112) 创建时反映 114 tasks + 12 paper-defense artifacts. Post-submission housekeeping (Tasks #113-#121) 完成后:
- 总 tasks: 114 → 121 (+7)
- verdicts/: 232 → 253 (+21: 11 verdicts + 10 data files)
- 新 paper-defense artifacts 0 → 11 (Task #113-#121 各一个)
- dispatcher 4 → 5 audits
- v1.0.0 tag 仍锚定 commit 9b81667

**Task #122 = TASKS_INDEX refresh** (mechanical update).

---

## 2. 实验设计 (writeup only)

### 2.1 TASKS_INDEX.md refresh

替换原 70 行文件为 ~110 行版本, 包含:
- Snapshot date: Task #112 → Task #121
- 3 artifact categories: 数字刷新 (116 → 123 descriptions, 232 → 253 verdicts, 46 products)
- Coverage matrix: 加 Task #113-#121 三行
- Recent closed tasks: Task #101-#121 (21 行表格, 之前只有 12)
- Paper-defense artifacts (10 件套, 不变)
- **新增** Post-submission housekeeping artifacts (11 件套, Tasks #111-#121)
- Conventions: 加 R9 + R8 §9.3 enforcement 描述

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| TASKS_INDEX.md 数字刷新 (114 → 121) | ✅ 闭环 |
| Coverage matrix 加 Task #113-#121 | ✅ 闭环 |
| Recent closed tasks 表格加 Task #113-#121 | ✅ 闭环 |
| 加 Post-submission housekeeping artifacts 段 | ✅ 闭环 |
| Conventions 段加 R8 §9.3 + R9 enforcement | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| TASKS_INDEX 长度 | ✅ ~110 行 (跟原版 70 行 + 增量 ~40 行) | ✅ 200+ 行 (过度细节) |
| 是否加 Post-submission housekeeping artifacts 表格 | ✅ 加 (跟 Paper-defense artifacts 表格 parallel 结构) | ❌ 不加 (但 reviewer 找不到 housekeeping 任务清单) |
| 是否更新 dispatcher audit count (4 → 5) | ✅ 更新 (Task #114 加了第 5 个 audit) | ❌ 保持 4 (但跟当前不一致) |
| 是否更新 Conventions 段 | ✅ 加 R8 §9.3 + R9 (Task #114 引入的 enforcement) | ❌ 保持原版 (但 reviewer 困惑 enforcement 来自哪) |
| 总 task 数 116 → 123 是否准确 | ✅ 121 numbered + 2 reference = 123 | ❌ 写 124 (overcount) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| 起草新 TASKS_INDEX.md | ~5 min |
| dispatcher 验证 | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~7 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: TASKS_INDEX.md 数字跟实际 verdicts 描述 count 不一致 (例如 descriptions/ 实际 123 但写 121)
  → **缓解**: 用 `ls descriptions/ | wc -l` 和 `ls verdicts/ | wc -l` 实测 cross-validate

**风险 2**: 改 TASKS_INDEX.md 破坏 Task #112 的 [v1.0.0] 段格式 (Keep-a-Changelog)
  → **缓解**: TASKS_INDEX 是独立文件, 跟 CHANGELOG.md 无关

---

## 7. 完成度跟踪

- [x] 写 descriptions/task122_tasks_index_refresh.md (本文件)
- [x] 重写 TASKS_INDEX.md (110 行版本)
- [x] dispatcher 5/5 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #112 (TASKS_INDEX 初始创建) + #121 (post-submission synthesis 串联)
- 后置: 无 (TASKS_INDEX refresh 是 documentation housekeeping 终点)

---

**核心交付**: TASKS_INDEX.md 刷新到 2026-07-24 (post #121). 总 tasks 121, verdicts 253, dispatcher 5 audits, 21 个 recent closed tasks (Task #101-#121), 11 个 post-submission housekeeping artifacts (Task #111-#121). reviewer 看 TASKS_INDEX 知道完整 task landscape.