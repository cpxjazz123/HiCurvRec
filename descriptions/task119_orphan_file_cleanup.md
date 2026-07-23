# Task #119 — Orphan File Cleanup (descriptions + top-level task data + scripts/ orphans)

> **任务目的**: 收尾 Task #117 housekeeping, 处理剩余 orphan 文件. 让 reviewer `git clone + checkout v1.0.0` 后 `git status` 进一步干净 (untracked 162 → ~140, orphan 文件清零).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #117 (.gitignore + CLAUDE.md + requirements.txt tracked) 把 untracked noise 从 2097 → 162 (-92%). 剩余 162 untracked 包含:
- src/ + configs/ + tools/ + external/ (上游 GRID clone 目录, reviewer 可见即可, 不动)
- **本次处理的 orphan**: descriptions/task116/117/118.md (3 description 漏 track) + task109/119/123/27 _*.json/csv/png (10 个 top-level 数据文件未进 verdicts/) + scripts/task*.ckpt (4 个 broken symlink) + scripts/task87*.bak (1 个 backup) + slurm-*.out (1 个空 log) + scripts/task388v5_verdict.md (1 个 misplaced verdict)

**Task #119 = 收尾 orphan cleanup**.

---

## 2. 实验设计 (housekeeping only)

### 2.1 Track 3 orphan description 文件

```bash
git add descriptions/task116_track_scripts_papers.md \
        descriptions/task117_gitignore_project_specific.md \
        descriptions/task118_reproduce_setup_section.md
```

**理由**: 3 个 description 是 reviewer-facing 文件, Task #116/#117/#118 各自闭环时漏了 `git add`. 跟 Task #115 precedent 一致 (Task #115 tracked 104 description files).

### 2.2 Move 10 top-level task data 文件 → verdicts/ + track

```bash
# task109 (3 files)
mv task109_correlation_summary.json verdicts/task109_phonism_sid_correlation_data.json
mv task109_knn_quality_table.csv verdicts/task109_phonism_sid_knn_quality_table.csv
mv task109_knn_vs_recall.png verdicts/task109_phonism_sid_knn_vs_recall.png

# task119 (2 files)
mv task119_mckg_space_summary.json verdicts/task119_3tokenizer_kNN_data.json
mv task119_mckg_space_table.csv verdicts/task119_3tokenizer_kNN_table.csv

# task123 (2 files)
mv task123_cross_space_correlation.json verdicts/task123_cross_space_correlation_data.json
mv task123_cross_space_table.csv verdicts/task123_cross_space_correlation_table.csv

# task27 (3 files)
mv task27_3tokenizer_summary.json verdicts/task27_neighborhood_quality_summary.json
mv task27_3tokenizer_table.csv verdicts/task27_neighborhood_quality_table.csv
mv task27_enhanced_stats.json verdicts/task27_neighborhood_quality_enhanced.json

git add verdicts/task109_* verdicts/task119_* verdicts/task123_* verdicts/task27_*
```

**理由**: verdicts/ 目录已有 task62_tc_zador.csv, task103_paper_claims_audit.csv precedent — top-level analysis data 应该归档到 verdicts/. PNG/CSV 是 paper-defense evidence (跟 verdict 文件同级).

### 2.3 Delete 4 broken .ckpt symlinks in scripts/

```bash
rm scripts/task1_v3.ckpt \
   scripts/task3_l3w512.ckpt \
   scripts/task3_l4w256.ckpt \
   scripts/task3_l5w256.ckpt
```

**理由**: 4 个 .ckpt 是 symlink 到 `/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/.../checkpoints/checkpoint_*.ckpt`. 这些是 Stage 1/2 训练时 debugger 临时创建的指向 GRID logs 的软链. GRID logs 不在 git tree (transient), symlink 在 repo 是 dead links. 应该删除.

### 2.4 Delete 1 orphan .bak in scripts/

```bash
rm scripts/task87_tiger_baseline_eval.py.bak.093854
```

**理由**: 备份文件, 跟 `scripts/task87_tiger_baseline_eval.py` 配套. 编辑脚本时自动生成的 .bak, 没用了. 删除.

### 2.5 gitignore slurm-*.out

```bash
# Append to .gitignore:
echo "" >> .gitignore
echo "# Slurm job stdout (transient cluster logs)" >> .gitignore
echo "slurm-*.out" >> .gitignore
```

**理由**: `slurm-58362148.out` (0 bytes) 是 HPC cluster job log. Slurm 输出不进 git (transient). 加 gitignore 防止以后再有 slurm-*.out 出现.

### 2.6 Move 1 misplaced verdict to verdicts/

```bash
mv scripts/task388v5_verdict.md verdicts/task388_v5_5defense_verdict.md
git add verdicts/task388_v5_5defense_verdict.md
```

**理由**: 文件首行 "task388v5 综合判定: 5 道防线版 H-E-E-E / H-H-E-E / H-H-H-H 最终结论" — 明确是 verdict 但位置错了. 应该进 verdicts/. verdicts/ 还没 task388* 文件 (确认过 `ls verdicts/task388*` 空).

---

## 3. 验证

```bash
# 1. untracked 减少
git ls-files --others --exclude-standard | wc -l   # 期望 162 → ~140

# 2. orphan 文件清零
ls scripts/task*.ckpt 2>&1           # 期望 No such file
ls scripts/*.bak 2>&1                # 期望 No such file
ls scripts/task388v5_verdict.md 2>&1 # 期望 No such file
ls slurm-*.out 2>&1                  # 期望 No such file (gitignored)

# 3. moved files in verdicts/
ls verdicts/task109_phonism_sid_*    # 期望 3 files
ls verdicts/task119_3tokenizer_kNN_* # 期望 2 files
ls verdicts/task123_cross_space_*    # 期望 2 files
ls verdicts/task27_neighborhood_quality_*  # 期望 3 files
ls verdicts/task388_v5_5defense_verdict.md # 期望 1 file

# 4. dispatcher 5/5 PASS
python3 scripts/all_audits.py
```

---

## 4. 决策触发

| 条件 | 决策 |
|------|------|
| 10 top-level 数据文件 mv 到 verdicts/ + tracked | ✅ 闭环 |
| 4 .ckpt symlink 删除 | ✅ 闭环 |
| 1 .bak 删除 | ✅ 闭环 |
| slurm-*.out gitignore | ✅ 闭环 |
| task388 verdict mv + tracked | ✅ 闭环 |
| 3 description tracked | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |
| untracked 162 → ~140 (-22) | ✅ 闭环 |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep + 起草 description | ~3 min |
| mv + git add + rm + .gitignore append | ~2 min |
| dispatcher 验证 | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~7 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: mv task388 verdict 后, 旧路径 scripts/task388v5_verdict.md 可能被某个旧脚本引用 (例如 `cat scripts/task388v5_verdict.md` in task111 synthesis). 但 task111 verdict 已写, 没引用旧路径. mv 安全.
  → **缓解**: mv 前 grep `scripts/task388v5_verdict.md` 检查引用. 没引用就 mv.

**风险 2**: mv top-level 数据文件后, 旧路径如果被 reference (例如某个 README 写 "see task109_correlation_summary.json"). 没找到 reference (经验上 README 不引用这些 top-level 数据).
  → **缓解**: mv 前 grep `task109_correlation_summary.json` 等 10 个文件名. 没引用就 mv.

**风险 3**: 删除 4 .ckpt symlink 后, 某些 R12 train 脚本 (`train_hrqvae.py`) 可能尝试 read 这些 symlink. 但 R12 用 `products/task<N>/checkpoints/`, 不在 scripts/. 安全.
  → **缓解**: 删除前 grep `scripts/task1_v3.ckpt` 等 4 个文件名. 没引用就 rm.

**风险 4**: .gitignore 改坏, 影响其他已 tracked 文件 (误 ignore). 
  → **缓解**: append 新规则, 不动现有. 跑 `git check-ignore -v <existing-tracked-file>` 验证.

---

## 7. 完成度跟踪

- [ ] grep reference 检查 (3 个 mv/rm 都先 grep)
- [ ] track 3 orphan description
- [ ] mv 10 top-level task data → verdicts/ + tracked
- [ ] rm 4 .ckpt symlinks
- [ ] rm 1 .bak
- [ ] gitignore slurm-*.out
- [ ] mv 1 misplaced verdict → verdicts/
- [ ] dispatcher 5/5 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #115 (track verdicts/descriptions) + #116 (track scripts/papers) + #117 (.gitignore + CLAUDE.md + requirements.txt tracked) + #118 (REPRODUCE.md §0)
- 后置: Task #120+ (src/ + configs/ + tools/ 解释文档 / 或 paper submission final cleanup)

---

**核心交付**: orphan file 全清理. untracked 162 → ~140 (-22). verdicts/ 多 11 个 paper-defense evidence 文件 (10 数据 + 1 verdict). descriptions/ 多 3 个 task description tracked.