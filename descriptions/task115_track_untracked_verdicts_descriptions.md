# Task #115 — Track Untracked Verdicts + Descriptions in Git

> **任务目的**: 把 166 untracked verdicts + 103 untracked descriptions 加进 git index. 让 reviewer 可以 `git checkout v1.0.0` 拿到完整 project history (verdicts + descriptions 全部 tracked), 不再依赖 working tree 中的 untracked files. 形成 R8/R9 transparency 闭环.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

初始 session-start git status 显示整个 `verdicts/` 和部分 `descriptions/` 是 untracked (`??`). 早期 task (#1-#100) 写 verdict 文件但未 `git add`. 这造成:
- reviewer `git clone` + `git checkout v1.0.0` 只能拿到 tracked 部分 (~69 verdicts, 11 descriptions), 看不全
- R8 §9.1 "verdicts/task<id>_result.md 必须存在" 看似满足, 实际本地有但 git 历史无
- R9 "descriptions/ task<N>_<slug>.md" 同理

Task #114 (5th audit) 顺带 tracked 47 verdict files (50 retroactive fix + 1 new task114 verdict). 现在还剩 166 verdicts + 103 descriptions untracked.

**Task #115 = untracked file commit**:
- `git add verdicts/*.md verdicts/*.json verdicts/*.csv verdicts/*.png verdicts/*.txt`
- `git add descriptions/task*.md descriptions/_general_pipeline.md`
- 单 commit 包含所有 untracked task-definition + verdict files
- 不动 `data/`, `logs/`, `task_artifacts/`, `papers/` (data + logs 通常 gitignored, papers 部分 tracked 部分 untracked 留待单独 task)
- 不动 `scripts/` (472 untracked 是 utility scripts, 不是 task deliverables)

---

## 2. 实验设计 (writeup only)

### 2.1 scope

| 目录 | untracked count | 决策 |
|------|-----------------|------|
| verdicts/ | 166 | ✅ 全 add (task deliverables) |
| descriptions/ | 103 | ✅ 全 add (task definitions, R9 强制) |
| scripts/ | 472 | ❌ 不动 (utility scripts, 单独 task #116+) |
| data/ | 152+152+152+24 | ❌ 不动 (dataset CSV, 通常 .gitignored) |
| logs/ | 145 | ❌ 不动 (运行日志, 通常 .gitignored) |
| task_artifacts/ | 26 | ❌ 不动 (临时产物, 单独 task) |
| papers/ | 21 | ⚠️ 部分 tracked, 部分 untracked (e.g., DECOR.pdf 已有但 paper.pdf tracked); 单独 task |

### 2.2 验证

```bash
# 1. 列出待 add 文件
git add verdicts/* descriptions/task* descriptions/_general_pipeline.md
git status --short | grep "^A" | wc -l
# 应 ~ 269 (166 + 103)

# 2. dispatcher 仍 5/5 PASS (不破坏)
python3 scripts/all_audits.py

# 3. R9 descriptions/ contiguous (A3 应仍 PASS)
python3 scripts/task114_verdict_integrity.py
```

### 2.3 commit message

```
task115: track 166 verdicts + 103 descriptions in git

- verdicts/: 166 previously untracked files now in git index
  (task10_phenom2_analysis, task100_paper_submission_prep_result,
   task116-134 diagnostic verdicts, etc.)
- descriptions/: 103 previously untracked task definitions
- reviewer can now `git checkout v1.0.0` and see complete history
- scripts/, data/, logs/, papers/ untouched (separate task #116+)
- dispatcher 5/5 PASS verified no regression
```

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| verdicts/* 全 add | ✅ 闭环 |
| descriptions/task* 全 add | ✅ 闭环 |
| descriptions/_general_pipeline.md add | ✅ 闭环 |
| 不 add scripts/, data/, logs/, papers/, task_artifacts/ | ✅ 闭环 |
| 单 commit (避免噪音) | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |
| 不动 verdict 内容 (仅 add) | ✅ 闭环 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| git add verdicts/* | ~5 sec |
| git add descriptions/* | ~5 sec |
| git status 验证 | ~2 sec |
| git commit (含 commit message) | ~10 sec |
| dispatcher 5/5 PASS 验证 | ~1 sec |
| 写 verdict + §16 + commit | ~3 min |
| **总计** | **~5 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: 269 文件 commit 信息密度过大, reviewer 难消化
  → **缓解**: 单 commit + 清晰 commit message 说明 ("166 verdicts + 103 descriptions")

**风险 2**: 部分 untracked verdicts/descriptions 实际是 obsolete 或 duplicate
  → **缓解**: 不在此 task 清理, 仅 add. 清理留 Task #116+ (orphan cleanup)

**风险 3**: descriptions/task1_, task2_, ... 文件名 pattern 不全 (有 task27_xxx + task27_yyy 各 2 个)
  → **缓解**: git add descriptions/* (glob) 自带 pattern, 不需手列

**风险 4**: verdicts/* 包含 .json, .csv, .png 等 binary, 不小心 commit 大文件
  → **缓解**: 验证 add 前文件大小 (`ls -la verdicts/*.png | sort -k5 -n`)

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task115_track_untracked_verdicts_descriptions.md` (本文件)
- [ ] `git add verdicts/*` (166 files)
- [ ] `git add descriptions/*` (103 files)
- [ ] `git status` 验证 add 正确
- [ ] dispatcher 5/5 PASS
- [ ] git commit
- [ ] loop.md §16 更新

---

## 7. 关联

- 前置: Task #114 (5th audit, tracked 47 verdicts)
- 后置: Task #116+ (track scripts/, data/, logs/, papers/ 单独处理)

---

**核心交付**: 269 untracked files (166 verdicts + 103 descriptions) tracked in git. reviewer 可 `git checkout v1.0.0` 拿完整 project history. R8/R9 transparency 闭环.