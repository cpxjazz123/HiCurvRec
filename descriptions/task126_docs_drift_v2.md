# Task #126 — TASKS_INDEX + CHANGELOG Documentation Drift v2

> **任务目的**: 修复 post-#121 housekeeping 之后新出现的 documentation drift: TASKS_INDEX.md coverage matrix 行只到 "Task #122 - #123", Recent closed tasks 标题写 "Task #101 - Task #121" 但 #124/#125 缺失; CHANGELOG [Unreleased] Notes 也只到 #121. 这些是 reviewer-facing drift, 必须补到跟实际 git 状态一致. 同步 dispatcher 5/5 PASS 验证.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #122 (TASKS_INDEX refresh) 写 snapshot 到 Task #121. 后续 #122/#123/#124/#125 4 个 housekeeping 任务已闭环, 但 TASKS_INDEX + CHANGELOG 没 refresh v2. 现在 review TASKS_INDEX.md / CHANGELOG.md 仍停在 #121 (即 4 个 post-#121 tasks 没有出现在 reviewer-facing 文档). 这是真实的 documentation drift, 跟 v1.0.0 已闭环状态不一致.

**Task #126 = docs drift v2 refresh** (post-#121 housekeeping 续集).

---

## 2. 实验设计 (writeup only)

### 2.1 修复 TASKS_INDEX.md coverage matrix

Add row:
```
| Task #124 - Task #125 | ✓ closed | README + TASKS_INDEX stale number refresh v2 + CI workflow dispatcher sync |
```

### 2.2 修复 TASKS_INDEX.md Recent closed tasks 标题 + 加 #124/#125 行

- 标题: `## Recent closed tasks (Task #101 - Task #121)` → `(Task #101 - Task #125)`
- Append 2 rows for #124 + #125

### 2.3 修复 CHANGELOG.md [Unreleased] Notes 段

Append 1 sentence:
```
- Continued housekeeping cycle (post-#121): #122 TASKS_INDEX refresh →
  #123 REPRODUCE.md script alignment → #124 README+TASKS_INDEX stale
  number refresh v2 → #125 CI workflow dispatcher sync.
```

### 2.4 验证 dispatcher 5/5 PASS

`python3 scripts/all_audits.py` → 5/5 PASS.

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| TASKS_INDEX coverage matrix + #124-#125 行 | ✅ 闭环 |
| TASKS_INDEX Recent closed tasks 标题 + 行 | ✅ 闭环 |
| CHANGELOG [Unreleased] Notes 段追加 | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| TASKS_INDEX 是否 refresh | ✅ Refresh v2 (cross-validation 必做) | ❌ 不动 (drift 持续扩大) |
| CHANGELOG [Unreleased] 是否补 #122-#125 | ✅ Append 一行 (Keep-a-Changelog 1.1.0 推荐) | ❌ 不补 (Unreleased 应跟当前状态一致) |
| 是否 gitignore src/configs/tools | ❌ 不动 (REPRODUCE §0 onboarding 期望 visible) | ✅ gitignore (但破坏 onboarding verify step) |
| README line 58 "125 entries" 是否改 | ❌ 不动 (125 仍是实际数) | ✅ 改成 128 (但 descriptions/ tracked 没变) |

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

**风险 1**: TASKS_INDEX 跟 CHANGELOG 数字不一致 (e.g. TASKS_INDEX 写 "Task #101 - #125" 但 CHANGELOG 没记录)
  → **缓解**: 两文档同时 refresh, 用 git log 作 single source of truth

**风险 2**: 加新行格式不匹配现有表格
  → **缓解**: 严格 copy 现有 #118/#119/#120 行格式 (Task | Subject | Status | Verdict)

---

## 7. 完成度跟踪

- [x] 写 descriptions/task126_docs_drift_v2.md (本文件)
- [x] TASKS_INDEX coverage matrix 加 #124-#125 行
- [x] TASKS_INDEX Recent closed tasks 标题 + #124/#125 行
- [x] CHANGELOG [Unreleased] Notes 追加 #122-#125
- [x] dispatcher 5/5 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #121 (post-submission housekeeping synthesis) + #122-#125 (4 post-#121 housekeeping tasks)
- 后置: 无 (post-#121 housekeeping cycle 续集闭环)

---

**核心交付**: TASKS_INDEX + CHANGELOG 同步到 Task #125 状态. coverage matrix + Recent closed tasks + [Unreleased] Notes 三处同步. dispatcher 5/5 PASS 不破坏.
