# Task #120 — gitignore Paper Reference Framework Clones (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: .gitignore append 13 rules 沉默 15 paper reference framework clones + reports/ + result/ + data/recbole/. untracked 143 → 114 (-29). 剩余 114 仅含 src/ (64) + configs/ (30) + tools/ (19) + 1 description — 全部 GRID 上游 framework (REPRODUCE.md §0 已说明). dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| 9 top-level paper clones gitignored | `git check-ignore -v BLOGER/README.md` | ✅ .gitignore:241 |
| /external/ gitignored | `git check-ignore -v external/DuoRec/README.md` | ✅ .gitignore:252 |
| reports/ gitignored | `git check-ignore -v reports/task22_pm_rq/phase1_toy_report.md` | ✅ .gitignore:255 |
| result/ gitignored | `git check-ignore -v result/task69_step1_global/baseline.json` | ✅ .gitignore:256 |
| data/recbole/ gitignored | `git check-ignore -v data/recbole/Musical_Instruments` | ✅ .gitignore:259 |
| src/ + configs/ + tools/ 仍 visible | `git check-ignore -v src/train.py configs/experiment/tiger_train_flat.yaml` | ✅ NOT ignored (无输出) |
| data/amazon_data/ 已 gitignore (Task #117) | `git check-ignore -v data/amazon_data/toys` | ✅ .gitignore:203 |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| untracked 143 → 114 | `git ls-files --others --exclude-standard \| wc -l` | ✅ -29 |

## 2. .gitignore append 13 rules

```gitignore
# === Paper reference framework clones (foreign git repos, Phase 4-7 baseline work) ===

# Top-level paper reference clones (~13 GB total, each has own .git/)
BLOGER/
DECOR/
ETEGRec/
HG-Rec/
LETTER/
RecBole/
RippleNet/
knowledge_graph_attention_network/
genrec/

# External framework sub-clones (paper baselines, ~15 GB)
/external/

# === Legacy analysis outputs (regeneratable, not reviewer-facing) ===
reports/
result/

# === Large binary data (RecBole framework data, 573M) ===
data/recbole/
```

## 3. untracked 143 → 114 breakdown

| 类别 | Before | After | 原因 |
|------|--------|-------|------|
| 9 top-level paper clones | ~9 dirs | 0 | gitignore |
| external/ (7 subdirs) | ~7 dirs | 0 | /external/ gitignore |
| reports/ | 2 dirs (44K) | 0 | reports/ gitignore |
| result/ | 3 dirs (48K) | 0 | result/ gitignore |
| data/recbole/ | 1 dir (573M) | 0 | data/recbole/ gitignore |
| src/ | 64 | 64 (unchanged) | GRID upstream, REPRODUCE.md §0 |
| configs/ | 30 | 30 (unchanged) | Hydra YAMLs |
| tools/ | 19 | 19 (unchanged) | Phase 7 migration |
| scripts/task87*.bak | 1 | 0 | rm -f |
| 1 description (task120) | 1 | 0 | git add (in commit) |
| **总 untracked** | **143** | **114** | **-29** |

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 9 top-level paper clones | ✅ gitignore (foreign git repos, ~13 GB, GeneRec 不该 track) | ❌ submodule (额外复杂度) |
| /external/ 处理 | ✅ gitignore (~15 GB, 7 sub-clones) | ❌ 单独 ignore each subdir (verbose) |
| reports/ 处理 | ✅ gitignore (legacy Phase 1-4, superseded by verdicts/) | ❌ track (但 7 reports 是旧分析, reviewer 不需要) |
| result/ 处理 | ✅ gitignore (transient analytical outputs) | ❌ mv verdicts/ (task69 verdicts 已存在) |
| data/recbole/ 处理 | ✅ gitignore (573M large binary) | ❌ track (太大, 无 git LFS) |
| src/ + configs/ + tools/ 处理 | ❌ KEEP visible (REPRODUCE.md §0 已说明 GRID 上游 cp -r) | ❌ gitignore (reviewer 困惑 "src 在哪") |
| 是否 track descriptions/task120 | ✅ track (本任务 description, reviewer 必看) | ❌ 不 track |
| scripts/task87*.bak | ✅ rm (orphan editor backup) | ❌ gitignore .bak (但 .bak 模式可能误伤其他) |

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| 9 top-level paper clones gitignored | `git check-ignore -v BLOGER/README.md` | ✅ line 241 |
| /external/ gitignored | `git check-ignore -v external/DuoRec/README.md` | ✅ line 252 |
| reports/ + result/ + data/recbole/ gitignored | check-ignore 4 paths | ✅ lines 255-259 |
| src/ + configs/ NOT ignored | check-ignore src/train.py + configs/.../tiger_train_flat.yaml | ✅ NOT matched |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| untracked 143 → 114 | `git ls-files --others --exclude-standard \| wc -l` | ✅ -29 |
| tracked count | `git ls-files \| wc -l` | ✅ 905 → 906 (+description) |

## 6. 剩余 114 untracked 的性质

| 类别 | 数量 | 性质 | reviewer 怎么处理 |
|------|------|------|------------------|
| src/ | 64 | GRID upstream framework (cp -r from GRID) | REPRODUCE.md §0 步骤复制 |
| configs/ | 30 | Hydra YAMLs from GRID | REPRODUCE.md §0 步骤复制 |
| tools/ | 19 | Phase 7 migration scripts | 可选, 跟 src/ 一起复制 |

**总评**: 剩余 114 untracked 全部是 GRID upstream framework, reviewer 按 REPRODUCE.md §0 三步 (clone GRID + cp -r 三目录 + verify) 即可补齐. 没有 "未解释的 orphan" 残留.

## 7. 关联

- 前置: Task #117 (.gitignore project-specific base) + #119 (orphan files cleanup)
- 后置: Task #121+ (剩余 114 untracked 仅 src/ + configs/ + tools/, 跟 REPRODUCE.md §0 配套; reviewer onboarding 最终闭环)

---

result: Task #120 — gitignore paper reference framework clones 闭环. .gitignore append 13 rules (9 top-level paper clones + /external/ + reports/ + result/ + data/recbole/). untracked 143 → 114 (-29). 剩余 114 全是 GRID upstream framework, REPRODUCE.md §0 闭环. dispatcher 5/5 PASS 不破坏.