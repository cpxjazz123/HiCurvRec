# Task #116 — Track Untracked Scripts + Reference Papers (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: 498 untracked files (466 utility scripts + 20 reference papers + 12 others) tracked in git. reviewer `git checkout v1.0.0` 现在拿完整 utility toolkit + reference paper corpus. dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| scripts/*.py + *.sh 全 add | `git ls-files --others --exclude-standard scripts/*.py scripts/*.sh \| wc -l` | ✅ 0 untracked remaining |
| papers/*.md + *.pdf + *.json 全 add | `git ls-files --others --exclude-standard papers/*.md papers/*.pdf papers/*.json \| wc -l` | ✅ 0 untracked remaining |
| 单 commit | `git log --oneline -1` | ✅ `89d9fb9 task116: track 466 utility scripts + 20 reference papers in git` |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| 不 add scripts/pids/, papers/notes/, papers/logs/ | `git status --short \| grep "^??"` | ✅ 这些目录仍 untracked (transient) |
| 不 add data/, logs/, task_artifacts/, products/ | `git status --short` | ✅ 仍 untracked (large binary) |
| 498 files tracked | `git diff --cached --stat` | ✅ 498 files |

## 2. 数字汇总

| 类别 | Before | After |
|------|--------|-------|
| tracked scripts | 152 (已 tracked 部分) | 618 (+466) |
| tracked papers | 6 (paper.md/pdf 已 tracked) | 26 (+20) |
| **总 tracked (scripts + papers)** | 158 | **644 (+486)** |

→ reviewer `git checkout v1.0.0` 现在拿到 644 utility/paper 文件 (vs 之前 158, +486 +307%).

## 3. tracked 内容覆盖

### 3.1 scripts/ tracked (466 files)

涵盖:
- Task #1 - #115 diagnostic scripts
- Stage 1/2/3/4 helpers (data prep, RQ-VAE, TIGER, evaluation)
- Phonism / HG-Rec / LETTER / TIGER / LightGCN baseline scripts
- Letter helpers + ablation scripts
- Codebook utilities (Sinkhorn, EMA, dead-code revival)
- HG-Rec curvature helpers (κ_M, free_curv)
- Phonism + Sinkhorn + Sinkhorn cascade
- Diagnostic scripts (stress-metric, Ollivier, neighborhood quality, etc.)
- Cleanup + audit scripts

### 3.2 papers/ tracked (20 files)

涵盖 8 paper reference corpus:
- BLOGER/ + DECOR/ + ETEGRec/ + HG-Rec/ + LETTER/ + MCKG/ + TIGER/ + latte_2605.06331/ + grid_paper.pdf
- 每个 .pdf 配 .md (extracted markdown) + .json (docling sidecar)

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 单 commit vs 多 commit | ✅ 单 commit (486+ files 同质 change: track untracked) | ❌ 多 commit (噪音) |
| scripts/pids/ 是否 track | ❌ 不 track (transient PID files) | ✅ track (但 review 看 stale PID 困惑) |
| papers/notes/ + papers/logs/ + papers/grid_paper.txt 是否 track | ❌ 不 track (transient 或中文 notes) | ✅ track (但增加 repo clutter) |
| data/ 是否 track (480 files CSV/tfrecord) | ❌ 不 track (large binary 通常 gitignored) | ✅ track (但 repo size 爆炸) |
| logs/ 是否 track (145 files) | ❌ 不 track (run logs, 通常 gitignored) | ✅ track (但 transient) |
| task_artifacts/ 是否 track (26 files) | ❌ 不 track (temp artifacts) | ✅ track (但 temp) |
| products/task88/ 是否 track (17 ckpt files) | ❌ 不 track (large ckpts, R12 mandate 已保存 1 个 ckpt) | ✅ track (但 size 爆炸) |
| commit message 详细度 | ✅ 详细 (按目录分组列出文件类型 + 决策 reason) | ❌ 简化 |

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| tracked scripts 数 | `git ls-files scripts/ \| wc -l` | ✅ 618 |
| tracked papers 数 | `git ls-files papers/ \| wc -l` | ✅ 26 |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| 5th audit A1 仍 117/117 | `python3 scripts/task114_verdict_integrity.py` | ✅ |
| total tracked (verdicts + descriptions + scripts + papers) | 350 + 644 = 994 files | ✅ |

## 6. 关联

- 前置: Task #115 (tracked 270 verdicts + descriptions)
- 后置: Task #117+ (data/ gitignore policy + scripts orphan cleanup)

---

result: Task #116 — track scripts + reference papers 闭环. 498 files 单 commit `89d9fb9` tracked in git: 466 utility scripts (task #1-115 diagnostic + Stage 1/2/3/4 helpers + HG-Rec/phonism/LETTER/TIGER/LightGCN baseline) + 20 reference papers (BLOGER/DECOR/ETEGRec/HG-Rec/LETTER/MCKG/TIGER/latte/grid_paper). reviewer 现在拿 644 scripts+papers 文件 (+486 vs before). scripts/pids/papers/notes/data/logs/task_artifacts/products 仍 untracked (transient 或 large binary). dispatcher 5/5 PASS 不破坏.