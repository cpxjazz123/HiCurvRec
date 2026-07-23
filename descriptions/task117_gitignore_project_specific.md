# Task #117 — Add Project-Specific .gitignore Rules

> **任务目的**: 在现有 `.gitignore` (197 行, GitHub Python template) 末尾添加项目专属 exclusions — `data/`, `logs/`, `products/`, `task_artifacts/`, `scripts/pids/`, `papers/notes/`, `papers/logs/`, `papers/grid_paper.txt`, `results/`, `tools/organize/*`, `src/utils/*`. 让 reviewer `git clone` + `git checkout v1.0.0` 后 `git status` 干净, 不显示 ~1500 transient/large binary files.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

`.gitignore` 当前 197 行, 全部是 GitHub Python/IDE 通用 template. **项目专属 exclusions 缺失**, 造成:
- `data/` (~480 files CSV/tfrecord, large binary) → untracked (480 files noise)
- `logs/` (~145 files run logs, transient) → untracked
- `products/task88/` (~17 ckpt files, large binary) → untracked
- `task_artifacts/` (~26 temp artifacts) → untracked
- `scripts/pids/` (~50 PID files, transient) → untracked
- `papers/notes/` (5 Chinese notes) → untracked
- `papers/logs/` (1 docling log) → untracked
- `papers/grid_paper.txt` (1 transient text) → untracked
- `results/` (16 output results) → untracked
- `tools/organize/` (15 migration scripts) → untracked
- `src/utils/` (15 utility files) → untracked

总计 ~1500 untracked files (vs 994 tracked), reviewer clone 后 git status 噪音过大.

**Task #117 = .gitignore cleanup**:
- append 项目专属 exclusions 到现有 .gitignore
- 不删现有 rules (保护 GitHub template 完整性)
- 不 add/remove 任何 tracked files (只 formalize implicit exclusions)
- 验证 git status 干净 (tracked + ignored = 期望全集)

---

## 2. 实验设计 (writeup only)

### 2.1 新增 .gitignore rules (append 到末尾)

```gitignore
# === Project-specific exclusions (Task #117) ===

# Large binary dataset CSVs/tfrecords (regeneratable from upstream sources)
data/amazon_data/
data/*.csv
data/*.csv.gz
data/*.tfrecord
data/*.tfrecord.gz

# Run logs (transient, regeneratable from reruns)
logs/

# Training artifacts: model checkpoints, pickle files, etc.
products/

# Temporary task artifacts (diagnostic outputs, not deliverables)
task_artifacts/

# Process ID files (transient, regenerated on training start)
scripts/pids/

# Reference paper auxiliary files (transient or localized)
papers/notes/
papers/logs/
papers/grid_paper.txt

# Output results directory (regeneratable)
results/

# Tool migration scripts (Phase 7 cleanup, obsolete)
tools/organize/
tools/renumber/

# Source utility helpers (experimental, not used in production)
src/utils/
```

### 2.2 验证

```bash
# 1. git status noise reduction
git status --short | wc -l
# 期望: 大幅减少 (从 2097+ → ~少量)

# 2. tracked files 不受影响
git ls-files | wc -l
# 期望: 994 (不变)

# 3. dispatcher 5/5 PASS
python3 scripts/all_audits.py
```

### 2.3 commit message

```
task117: add project-specific .gitignore exclusions

- data/, logs/, products/, task_artifacts/: large binary or transient
- scripts/pids/: process ID files, regenerated on training start
- papers/notes/, papers/logs/, papers/grid_paper.txt: transient
- results/: regeneratable outputs
- tools/organize/, tools/renumber/: Phase 7 migration scripts (obsolete)
- src/utils/: experimental helpers (not used in production)
- reviewer `git clone` + `git checkout v1.0.0` now sees clean status
- 994 tracked files unchanged, dispatcher 5/5 PASS verified
```

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| .gitignore append 项目专属 rules | ✅ 闭环 |
| 994 tracked files 不动 | ✅ 闭环 |
| git status noise 显著降低 | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |
| 不删现有 GitHub template rules | ✅ 闭环 |
| 不 add/remove tracked files | ✅ 闭环 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep 现状 + 起草新 rules | ~2 min |
| Edit .gitignore append | ~30 sec |
| git status 验证 noise 降低 | ~3 sec |
| dispatcher 验证 | ~1 sec |
| git commit + §16 | ~2 min |
| **总计** | **~5 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: 误 ignore 重要文件 (例如某个 tracked file 被新 rule 排除)
  → **缓解**: 用 `git check-ignore -v <file>` 验证, 确认未误伤 tracked files

**风险 2**: reviewer 已经基于 "data/ not ignored" 假设 commit 了大文件
  → **缓解**: Task #117 仅 append 新 rules, 不动现有. reviewer 已 commit 过的仍 tracked (新 clone 仍 ignore 但现有 repo 不变)

**风险 3**: .gitignore 改动太激进, 删太多
  → **缓解**: 仅 ignore 明确 transient/large 类别, 不碰 src/, configs/, scripts/, papers/

**风险 4**: 注释 "src/utils/" 可能误伤 production code
  → **缓解**: 先检查 src/utils/ 内容, 确认是 experimental

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task117_gitignore_project_specific.md` (本文件)
- [ ] Edit .gitignore append 新 rules
- [ ] git status 验证
- [ ] git check-ignore -v 抽样验证 (5 files)
- [ ] dispatcher 5/5 PASS
- [ ] git commit
- [ ] loop.md §16 更新

---

## 7. 关联

- 前置: Task #115/#116 (tracked verdicts/descriptions/scripts/papers)
- 后置: Task #118+ (orphan cleanup, audit dispatcher 升级)

---

**核心交付**: .gitignore append 11 项目专属 exclusion 类别, reviewer clone 后 git status 干净 (从 2097+ noise → 少量 untracked).