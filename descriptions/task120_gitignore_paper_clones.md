# Task #120 — gitignore Paper Reference Framework Clones + reports/result/data:recbole

> **任务目的**: 收尾 orphan cleanup. 15 个 paper reference framework clones (foreign git repos, 共 ~28 GB) + reports/ + result/ + data/recbole/ 都该 gitignore. 让 reviewer `git clone + checkout v1.0.0` 后 `git status` 仅显示 GRID 上游 framework (src/ + configs/ + tools/ + data/amazon_data) 必需的 imported content. untracked 143 → ~115.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #119 把 untracked 162 → 143 (-19), 处理了 orphan description/data/scripts files. 剩余 143 untracked 包含:

**Framework clones (foreign git repos, 应该 gitignore)**:
- 9 个 top-level 论文 reference clones (~13 GB): BLOGER (53M), DECOR (2.5G), ETEGRec (3.8G), HG-Rec (124M), LETTER (570M), RecBole (5.5G), RippleNet (374M), knowledge_graph_attention_network/KGAT (464M), genrec (7.4M)
- 7 个 external/ 子目录 (~15 GB): DuoRec, ETEGRec, FMLP-Rec, LETTER, LLM-RecSys-ID, P5, checkpoint

**Legacy analysis outputs (应该 gitignore 或 track?)**:
- reports/ (44K, 7 markdown reports from old Phase 1-4 work: task22_pm_rq + task99_mckg_rebuild)
- result/ (48K, 3 JSON outputs from task69_step1_global/baseline.json + step23_conditional/result.json + step45_verdict)

**Large binary data (应该 gitignore)**:
- data/recbole/ (573M, RecBole framework data)

**Keep visible (GRID 上游 framework, reviewer 必需)**:
- src/ (64 files, 上游 GRID pipeline 代码, REPRODUCE.md §0 已说明)
- configs/ (30 files, Hydra YAMLs)
- tools/ (19 files, Phase 7 migration scripts)
- data/amazon_data/ (5.8G, 已部分 gitignore, 数据集)

**Task #120 = gitignore 15 paper clones + reports/ + result/ + data/recbole/**

---

## 2. 实验设计 (housekeeping only)

### 2.1 Append .gitignore rules

```gitignore
# === Paper reference framework clones (foreign git repos with own .git) ===

# Top-level paper reference clones (Phase 4-7 baseline reproduction work)
BLOGER/
DECOR/
ETEGRec/
HG-Rec/
LETTER/
RecBole/
RippleNet/
knowledge_graph_attention_network/
genrec/

# External framework clones (sub-clones of paper baselines)
/external/

# === Legacy analysis outputs (regeneratable from scripts) ===
reports/
result/

# === Large binary data (regeneratable from upstream sources) ===
data/recbole/
```

### 2.2 验证

```bash
# 1. 15 paper clones gitignored
git check-ignore -v BLOGER/ DECOR/ external/DuoRec external/ETEGRec 2>&1 | head -10

# 2. reports/ + result/ gitignored
git check-ignore -v reports/task22_pm_rq reports/task99_mckg_rebuild result/task69_step1_global 2>&1

# 3. data/recbole gitignored
git check-ignore -v data/recbole 2>&1

# 4. KEEP src/ + configs/ + tools/ + data/amazon_data/ visible (sanity)
git check-ignore -v src/train.py configs/experiment/tiger_train_flat.yaml data/amazon_data/toys/ 2>&1
# 应该 NOT match (即 not ignored)

# 5. dispatcher 5/5 PASS
python3 scripts/all_audits.py

# 6. untracked 143 → ~115
git ls-files --others --exclude-standard | wc -l
```

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| 9 top-level paper clones gitignored | ✅ 闭环 |
| /external/ gitignored | ✅ 闭环 |
| reports/ gitignored | ✅ 闭环 |
| result/ gitignored | ✅ 闭环 |
| data/recbole/ gitignored | ✅ 闭环 |
| src/ + configs/ + tools/ + data/amazon_data/ 仍 visible | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |
| untracked 143 → ~115 | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| reports/ 是否 track | ❌ gitignore (legacy Phase 1-4 reports, 已 superseded by verdicts/) | ✅ track (但 7 reports 是旧分析, reviewer 不需要) |
| result/ 是否 track | ❌ gitignore (transient analytical outputs, scripts 重新跑会生成) | ✅ mv 到 verdicts/ (但 task69 verdicts 已存在, result/ 是中间产物) |
| data/recbole/ 是否 track | ❌ gitignore (573M 大 binary, RecBole framework data) | ✅ track (但太大, git LFS 未配) |
| src/ + configs/ + tools/ 是否 gitignore | ❌ KEEP visible (REPRODUCE.md §0 已说明 GRID 上游 cp -r, reviewer 必需) | ✅ gitignore (但 reviewer 困惑 "src 在哪") |
| 9 top-level paper clones 是否 gitignore | ✅ gitignore (foreign git repos, GeneRec 不该 track) | ❌ submodule (但 git submodule 引入额外复杂度) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep 引用检查 | ~1 min |
| .gitignore append | ~30 sec |
| check-ignore 验证 | ~10 sec |
| dispatcher 验证 | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~4 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: gitignore reports/ + result/ 后, scripts 仍写这些路径. 但 scripts 跑会重新生成文件, gitignore 只是 hide from git status, 不影响 script 执行. 安全.

**风险 2**: data/recbole/ 是 RecBole framework 的 data, gitignore 后 reviewer 看不到. 但 RecBole 不在主 pipeline 里 (主 pipeline 用 GRID framework + amazon_data), 它的 data 不影响 paper reproduction. 安全.

**风险 3**: /external/ gitignore 后, 7 个子目录都被隐藏. external/checkpoint/ 没 .git 但跟 external/ 同源 (paper baselines). 一起 gitignore 简化.

**风险 4**: src/ + configs/ + tools/ + data/amazon_data/ 仍 visible, 但 reviewer `git clone` 后这些目录空 (因为 GRID 上游 cp -r 没在 git 里). REPRODUCE.md §0 已说明 reviewer 需要 cp -r 手动复制. 闭环.

---

## 7. 完成度跟踪

- [ ] grep 引用检查
- [ ] .gitignore append 15 paper clones + reports/ + result/ + data/recbole/
- [ ] check-ignore 验证
- [ ] dispatcher 5/5 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #117 (.gitignore project-specific) + #119 (orphan files cleanup)
- 后置: Task #121+ (剩余 ~115 untracked 仅 src/ + configs/ + tools/ + data/amazon_data/, 跟 REPRODUCE.md §0 配套; 或 paper submission final cleanup)

---

**核心交付**: .gitignore append 13 rules (9 top-level paper clones + /external/ + reports/ + result/ + data/recbole/). untracked 143 → ~115. reviewer `git clone` 后看到清晰的 GRID 上游 framework + GeneRec paper-defense 主体.