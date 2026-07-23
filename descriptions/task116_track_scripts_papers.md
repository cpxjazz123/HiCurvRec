# Task #116 — Track Untracked Scripts + Reference Papers in Git

> **任务目的**: 把 466 untracked utility scripts (*.py + *.sh) + 20 untracked reference papers (DECOR/ETEGRec/HG-Rec/MCKG/TIGER/latte .md/.pdf/.json) tracked in git. 让 reviewer 看到完整 utility toolkit + 8 paper reference corpus (BLOGER + DECOR + ETEGRec + HG-Rec + LETTER + RecBole + RippleNet + latte + MCKG + TIGER + grid_paper). 不 track pids/ / notes/ / logs/ / data/ / products/ (transient 或 large binary).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #115 tracked verdicts/ + descriptions/. 剩余 untracked 大类:
- `scripts/` 466 untracked utility scripts (*.py + *.sh, 不含 pids/)
- `papers/` 20 untracked reference paper corpus files
- `data/` ~480 untracked CSVs (large, 通常 gitignored)
- `logs/` 145 untracked run logs (transient, 通常 gitignored)
- `task_artifacts/` 26 untracked temp artifacts (transient)

**Task #116 = scripts + papers tracking**:
- scripts/*.py + scripts/*.sh (466 files, utility toolkit)
- papers/*.md + papers/*.pdf + papers/*.json (20 files, reference papers)
- 单 commit (`Reviewer transparency for utility toolkit + reference papers`)
- dispatcher 5/5 PASS 验证不破坏
- 不动 data/, logs/, pids/, task_artifacts/, products/

---

## 2. 实验设计 (writeup only)

### 2.1 scope

| 类别 | untracked count | 决策 | 原因 |
|------|-----------------|------|------|
| scripts/*.py + scripts/*.sh | 466 | ✅ track | utility scripts, reviewer 想看 diagnostic tools |
| scripts/pids/ | ~50 | ❌ 不 track | PID files, transient process tracking |
| papers/*.md + *.pdf + *.json | 20 | ✅ track | reference papers (BLOGER/DECOR/ETEGRec/HG-Rec/LETTER/MCKG/TIGER/latte), reviewer 需 reference corpus |
| papers/notes/ | 5 | ❌ 不 track | 中文 markdown notes, 不适合 reviewer |
| papers/logs/ | 1 | ❌ 不 track | docling logs, transient |
| papers/grid_paper.txt | 1 | ❌ 不 track | text extraction, transient |
| data/amazon_data/* | 480 | ❌ 不 track | large CSV/tfrecord binary, 通常 gitignored |
| logs/ | 145 | ❌ 不 track | run logs, transient |
| task_artifacts/ | 26 | ❌ 不 track | temp artifacts |
| products/task88/ | 17 | ❌ 不 track | ckpt files, large binary |

### 2.2 验证

```bash
# 1. stage 466 scripts + 20 papers
git add 'scripts/*.py' 'scripts/*.sh'
git add 'papers/*.md' 'papers/*.pdf' 'papers/*.json'
# Total ~ 486 files

# 2. dispatcher 5/5 PASS
python3 scripts/all_audits.py

# 3. R8/R9 不破坏 (A1, A3 应仍 PASS)
python3 scripts/task114_verdict_integrity.py
```

### 2.3 commit message

```
task116: track 466 utility scripts + 20 reference papers in git

- scripts/*.py + scripts/*.sh (466 utility scripts for diagnostics)
- papers/*.md + *.pdf + *.json (20 reference papers:
  BLOGER, DECOR, ETEGRec, HG-Rec, LETTER, MCKG, TIGER, latte, grid_paper)
- reviewer can now git checkout v1.0.0 and see full utility toolkit + reference corpus
- pids/, notes/, data/, logs/, task_artifacts/, products/ untouched (transient or large)
- dispatcher 5/5 PASS verified no regression
```

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| scripts/*.py + *.sh 全 add | ✅ 闭环 |
| papers/*.md + *.pdf + *.json 全 add | ✅ 闭环 |
| 不 add scripts/pids/, papers/notes/, papers/logs/ | ✅ 闭环 |
| 不 add data/, logs/, task_artifacts/, products/ | ✅ 闭环 |
| 单 commit | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| git add scripts/*.py scripts/*.sh | ~5 sec |
| git add papers/*.md papers/*.pdf papers/*.json | ~3 sec |
| git status 验证 | ~2 sec |
| git commit | ~5 sec |
| dispatcher 5/5 PASS 验证 | ~1 sec |
| 写 verdict + §16 + commit | ~3 min |
| **总计** | **~5 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: 466 scripts + 20 papers = 486 files commit 信息密度大
  → **缓解**: 单 commit + 详细 commit message (按目录分组列出)

**风险 2**: scripts/pids/ 包含过期的 PID 文件, commit 后会触发 "stale PID" 警告
  → **缓解**: 用 glob `scripts/*.py scripts/*.sh` 排除 pids/ 子目录

**风险 3**: papers/ 包含 ~3.2 MB HG-Rec.pdf, repo size 增加
  → **缓解**: 可接受 (paper reference corpus 是 reviewer-facing context), 比 data/ 小得多

**风险 4**: 一些 scripts 是早期 obsolete diagnostic, commit 后 reviewer 看 confusion
  → **缓解**: 不在此 task 清理, 仅 add. 清理留 Task #117+ (orphan scripts cleanup)

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task116_track_scripts_papers.md` (本文件)
- [ ] `git add scripts/*.py scripts/*.sh` (466 files)
- [ ] `git add papers/*.md papers/*.pdf papers/*.json` (20 files)
- [ ] `git status` 验证 add 正确
- [ ] dispatcher 5/5 PASS
- [ ] git commit
- [ ] loop.md §16 更新

---

## 7. 关联

- 前置: Task #115 (tracked 270 verdicts + descriptions)
- 后置: Task #117+ (scripts orphan cleanup + data/ gitignore policy)

---

**核心交付**: 486 untracked files (466 utility scripts + 20 reference papers) tracked in git. reviewer 拿完整 utility toolkit + reference paper corpus. R8/R9 transparency 进一步闭环.