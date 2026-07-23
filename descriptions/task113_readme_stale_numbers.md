# Task #113 — README.md Stale Number Refresh

> **任务目的**: 修复 `README.md` 中两处与 v1.0.0 release artifacts 不一致的数字: (1) line 58 "Task definitions (101 entries)" → 实际 114 task files; (2) line 61 "~480 files" → 实际 483 scripts/* files. 形成 reviewer 准确的入口.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

v1.0.0 release artifacts 已就位 (Task #110/112), 但 `README.md` §1 "What's in this repository" 段仍含 2026-07-22/23 时点的过时数字:
- line 58: `Task definitions (101 entries, R9 continuous)` → 实际 `descriptions/` 含 116 文件 = 114 task files (含 task27/task67 各 2 个版本) + 2 non-task (README.md, _general_pipeline.md)
- line 61: `Diagnostic & analysis scripts (~480 files)` → 实际 483 files = 318 .py + 154 .sh + 11 dirs/other

不动 line 33 `papers/paper.pdf (14 pages, 106 KB)` (验证正确, papers/paper.pdf = 106816 bytes).

**Task #113 = README 数字刷新**:
- 改 2 处数字 (101 → 114, 480 → 483)
- 不改结构, 不动 paper 内容, 不创建新 README (Rule 4)

---

## 2. 实验设计 (writeup only)

### 2.1 待改数字

| Line | 当前 | 实际 | 来源 |
|------|------|------|------|
| 58 | `Task definitions (101 entries, R9 continuous)` | 114 task files (含 #27/#67 各 2 版本) | `ls descriptions/ \| grep "^task" \| wc -l` |
| 61 | `Diagnostic & analysis scripts (~480 files)` | 483 files | `ls scripts/ \| wc -l` |

### 2.2 不改

- Line 33 `papers/paper.pdf (14 pages, 106 KB)` — 验证正确
- Line 42 `paper.pdf # Submission-ready PDF (14 pages)` — 正确
- Line 50-56 file structure (paths) — 全部正确
- §2 Headline Findings 数字 (R@10 0.1058/0.1051) — 与 verdict 一致
- §3-§7 内容 — 与 verdicts 一致
- 7 shields.io badges (Task #109) — 正确

### 2.3 验证

```bash
# 1. 检查 README 不再含 101 / 480 等过时数字
grep -nE "101 entries|480 files" README.md

# 2. 检查新数字正确
grep -nE "114 entries|483 files|318 \.py" README.md

# 3. dispatcher 仍 4/4 PASS (不破坏)
python3 scripts/all_audits.py
```

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| README line 58 含 `114 entries` | ✅ 闭环 |
| README line 61 含 `~483 files` | ✅ 闭环 |
| dispatcher 仍 4/4 PASS | ✅ 闭环 |
| README 长度 < 220 行 (小幅改) | ✅ 闭环 |
| 不创建新 README | ✅ 闭环 (Rule 4) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep 找 stale 数字 | ~1 min |
| 改 2 处数字 | ~2 min |
| 最终验证 (grep + dispatcher) | ~2 min |
| git commit + §16 更新 | ~2 min |
| **总计** | **~7 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: 改了数字但 break other sections
  → **缓解**: 只改 line 58 和 61, 不动其他

**风险 2**: 数字改了但仍 stale (e.g., script count 变化)
  → **缓解**: 用 `ls scripts/ | wc -l` 现算, 不凭记忆

**风险 3**: README.md 不在 tracked files (history 显示 untracked)
  → **缓解**: 验证 `git log -- README.md` 确认 tracked

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task113_readme_stale_numbers.md` (本文件)
- [ ] 改 README line 58 (`101 entries` → `114 entries`)
- [ ] 改 README line 61 (`~480 files` → `~483 files`)
- [ ] 最终验证 (grep + dispatcher)
- [ ] git commit
- [ ] loop.md §16 更新

---

## 7. 关联

- 前置: Task #112 (v1.0.0 release artifacts)
- 后置: README.md 与 v1.0.0 release artifacts 完全一致

---

**核心交付**: README.md 2 处 stale 数字刷新, 与实际文件数对齐. v1.0.0 release artifacts 一致性闭环.