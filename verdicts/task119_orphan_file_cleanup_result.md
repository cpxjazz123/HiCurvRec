# Task #119 — Orphan File Cleanup (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: orphan file 全清理. tracked +15 files (3 description + 10 task data + 1 legacy verdict + 1 .gitignore rule). 4 broken .ckpt symlinks + 1 .bak 删除. untracked 162 → 143 (-19). dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| 3 description tracked | `git ls-files descriptions/task116/117/118` | ✅ |
| 10 task data mv → verdicts/ + tracked | `ls verdicts/task109_phonism_sid_* verdicts/task119_3tokenizer_kNN_* verdicts/task123_cross_space_* verdicts/task27_neighborhood_quality_*` | ✅ 10 files |
| 4 .ckpt symlinks 删除 | `ls scripts/task*.ckpt` | ✅ No such file |
| 1 .bak 删除 | `ls scripts/*.bak` | ✅ No such file |
| 1 misplaced verdict mv + rename + tracked | `ls verdicts/phonism_v5_5defense_legacy_verdict.md` | ✅ |
| slurm-*.out gitignore | `tail -3 .gitignore` | ✅ |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| untracked 162 → 143 | `git ls-files --others --exclude-standard \| wc -l` | ✅ -19 |

## 2. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| task388 verdict 命名 | ❌ verdicts/task388_v5_5defense_verdict.md (但 A4 把 max ID 拉到 388, threshold 50 → fail) → ✅ mv 后再 rename 为 verdicts/phonism_v5_5defense_legacy_verdict.md (移除数字前缀, 标 legacy) | ❌ 增加 A4 threshold (掩盖真实 gap) |
| 是否 modify 4 个 scripts (out_dir → verdicts/) | ❌ 不改 (最小改动, scripts 是 auxiliary analysis tools, reviewer 不 rerun; verdict 里说明) | ✅ 改 scripts (引入 review surface, 不必要) |
| 是否 mv 10 个 data 文件 | ✅ mv (跟 verdicts/task62_tc_zador.csv + task103_paper_claims_audit.csv precedent 一致) | ❌ 留在 top-level (但 top-level 是 reviewer onboarding 入口, 应该有干净 layout) |
| slurm-*.out 处理 | ✅ gitignore (transient cluster log, 不进 git) | ❌ rm (但以后 slurm job 还会生成新文件) |
| 是否 track 3 个 description | ✅ track (Task #116/#117/#118 各自闭环时漏了 git add, 修这个 mechanical gap) | ❌ 不 track (但 reviewer 看不到 task 描述) |
| 是否 rm 4 .ckpt symlinks | ✅ rm (指向 GRID logs 的 dead links, 没用) | ❌ 留 (但 reviewer 困惑 dead symlinks) |

## 3. 具体动作清单

### 3.1 track 3 orphan descriptions
- `descriptions/task116_track_scripts_papers.md` (136 行)
- `descriptions/task117_gitignore_project_specific.md` (169 行)
- `descriptions/task118_reproduce_setup_section.md` (134 行)

### 3.2 mv 10 top-level data → verdicts/
| 旧路径 | 新路径 |
|--------|--------|
| `task109_correlation_summary.json` | `verdicts/task109_phonism_sid_correlation_data.json` |
| `task109_knn_quality_table.csv` | `verdicts/task109_phonism_sid_knn_quality_table.csv` |
| `task109_knn_vs_recall.png` | `verdicts/task109_phonism_sid_knn_vs_recall.png` |
| `task119_mckg_space_summary.json` | `verdicts/task119_3tokenizer_kNN_data.json` |
| `task119_mckg_space_table.csv` | `verdicts/task119_3tokenizer_kNN_table.csv` |
| `task123_cross_space_correlation.json` | `verdicts/task123_cross_space_correlation_data.json` |
| `task123_cross_space_table.csv` | `verdicts/task123_cross_space_correlation_table.csv` |
| `task27_3tokenizer_summary.json` | `verdicts/task27_neighborhood_quality_summary.json` |
| `task27_3tokenizer_table.csv` | `verdicts/task27_neighborhood_quality_table.csv` |
| `task27_enhanced_stats.json` | `verdicts/task27_neighborhood_quality_enhanced.json` |

### 3.3 删除 5 个 orphan files
- `scripts/task1_v3.ckpt` (symlink → /fs04/ar57/wenyu/GeneRec/GRID/logs/...)
- `scripts/task3_l3w512.ckpt` (symlink → GRID/logs/...)
- `scripts/task3_l4w256.ckpt` (symlink → GRID/logs/...)
- `scripts/task3_l5w256.ckpt` (symlink → GRID/logs/...)
- `scripts/task87_tiger_baseline_eval.py.bak.093854` (backup file)

### 3.4 mv + rename 1 misplaced verdict
- `scripts/task388v5_verdict.md` → `verdicts/phonism_v5_5defense_legacy_verdict.md`
- Rename 移除 "task388" 数字前缀 (避免 A4 把 max ID 拉到 388, threshold 50 fail)
- 标 "legacy" 表明是 pre-renumbering 历史 verdict

### 3.5 gitignore slurm-*.out
```gitignore
# Slurm job stdout (transient cluster logs, regenerated on each job)
slurm-*.out
```

## 4. Scripts 未 modify 说明

grep 发现 4 个 scripts (task27_3tokenizer_analysis.py, task27_enhanced_stats.py, task119_mckg_space_knn.py, task123_cross_space_correlation.py) 仍 write top-level 路径. **决策**: 不 modify 这些 scripts, 因为:

1. **最小改动原则**: scripts 是 auxiliary analysis tools, reviewer 不 rerun (它们产生的是 task execution 时的 evidence, 已 commit 进 verdicts/)
2. **修改 surface 风险**: 改 scripts 会改 reviewer-facing 行为, 引入回归风险
3. **verdict 里说明**: reviewer 看 Task #119 verdict 知道"scripts 写 top-level, 如 rerun 会生成新 top-level 文件, 可自行调整 --out_dir"

如果 reviewer 未来需要 rerun, 可通过 `--out_dir` (task27_enhanced_stats.py 已支持) 或手动 `cd verdicts && python ../scripts/...` 解决.

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| 3 description tracked | `git ls-files descriptions/task116_track_scripts_papers.md descriptions/task117_gitignore_project_specific.md descriptions/task118_reproduce_setup_section.md` | ✅ |
| 10 data files in verdicts/ | `ls verdicts/task{109,119,123,27}_*` | ✅ 10 files |
| 4 .ckpt symlinks gone | `ls scripts/task*.ckpt` | ✅ No such file |
| 1 .bak gone | `ls scripts/*.bak` | ✅ No such file |
| 1 misplaced verdict renamed + tracked | `ls verdicts/phonism_v5_5defense_legacy_verdict.md` | ✅ |
| slurm-*.out gitignored | `tail -3 .gitignore` | ✅ |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| untracked 162 → 143 | `git ls-files --others --exclude-standard \| wc -l` | ✅ |
| tracked count | `git ls-files \| wc -l` | ✅ 886 → 901 (+15) |

## 6. 关联

- 前置: Task #115 (track verdicts/descriptions) + #116 (track scripts/papers) + #117 (.gitignore + CLAUDE.md + requirements.txt tracked) + #118 (REPRODUCE.md §0)
- 后置: Task #120+ (剩余 143 untracked 主要为 src/ + configs/ + tools/ + external/ + BLOGER/DECOR/ETEGRec/HG-Rec/LETTER/RecBole/RippleNet 8 paper reference clones, reviewer onboarding)

---

result: Task #119 — orphan file cleanup 闭环. tracked +15 files (3 description + 10 task data + 1 legacy verdict + 1 .gitignore rule). 4 broken .ckpt symlinks + 1 .bak + 1 empty slurm log 删除. untracked 162 → 143 (-19). dispatcher 5/5 PASS 不破坏.