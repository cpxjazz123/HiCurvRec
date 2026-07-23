# Task #117 — Project-Specific .gitignore (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: `.gitignore` append 11 项目专属 exclusion rules + track `CLAUDE.md` + `requirements.txt`. reviewer `git clone` 后 untracked noise 从 2097 → 162 (-92%). tracked files 从 884 → 886 (+CLAUDE.md + requirements.txt). dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| .gitignore append 11 项目专属 rules | tail -25 .gitignore | ✅ |
| CLAUDE.md + requirements.txt tracked | `git ls-files CLAUDE.md requirements.txt \| wc -l` | ✅ 2 |
| Untracked noise 显著降低 | `git ls-files --others --exclude-standard \| wc -l` | ✅ 2097 → 162 (-92%) |
| Tracked 不变 (除 +2) | `git ls-files \| wc -l` | ✅ 884 → 886 (+CLAUDE.md + requirements.txt) |
| 抽样 check-ignore 验证 | `git check-ignore -v <file>` | ✅ data/ + scripts/pids/ + results/ 正确 ignore |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |
| 不动现有 GitHub template rules | git diff .gitignore | ✅ 仅 append, 不删 |

## 2. .gitignore append 11 rules

```gitignore
# === Project-specific exclusions (Task #117) ===

# Large binary dataset CSVs / tfrecords (regeneratable from upstream sources)
data/amazon_data/
data/*.csv
data/*.csv.gz
data/*.tfrecord
data/*.tfrecord.gz

# Run logs (transient, regeneratable from reruns)
logs/
log_tensorboard/

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

# Claude session data (settings, locks, scheduled tasks)
.claude/

# Project root marker (used by tooling, not a deliverable)
.project-root
```

## 3. untracked noise 降低

| 类别 | Before | After |
|------|--------|-------|
| 总 untracked | 2097 | 162 (-92%) |
| data/ | 480 | ~0 (已 ignore) |
| logs/ | 145 | 0 (已 ignore) |
| products/ | 17 | 0 (已 ignore) |
| task_artifacts/ | 26 | 0 (已 ignore) |
| scripts/pids/ | ~50 | 0 (已 ignore) |
| papers/notes/ + papers/logs/ + papers/grid_paper.txt | 7 | 0 (已 ignore) |
| results/ | 16 | 0 (已 ignore) |
| log_tensorboard/ | ~6 | 0 (已 ignore) |
| .claude/ | 7 | 0 (已 ignore) |
| .project-root | 1 | 0 (已 ignore) |

剩余 162 untracked 主要是: src/ (上游 GRID clone) + configs/ (Hydra YAMLs) + tools/ (Phase 7 migration scripts) + 几个 top-level task109/119/123/27 _*.json/csv/png (task supplementary data, 待 Task #118+ 决定).

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否 gitignore src/ | ❌ 不 ignore (上游 GRID clone, reviewer 需看到) | ✅ ignore (但 review 困惑"src 在哪") |
| 是否 gitignore configs/ | ❌ 不 ignore (Hydra YAMLs, reviewer 需看到) | ✅ ignore (但 Hydra 默认从 configs/ 读) |
| 是否 gitignore tools/ | ❌ 不 ignore (Phase 7 工具脚本, 可见即可) | ✅ ignore (但 review 不知道有 tools/) |
| 是否 gitignore external/ | ❌ 不 ignore (12 个 reference framework clone, 可见即可) | ✅ ignore |
| 是否 track CLAUDE.md | ✅ track (项目指令, reviewer 必看) | ❌ untracked (但 review 困惑"AI 怎么用") |
| 是否 track requirements.txt | ✅ track (Python deps, reproducibility 关键) | ❌ untracked (但 review 困惑"装什么包") |
| 是否 track top-level task109/119/123/27 _*.json/csv/png | ❌ 不在此 task (需单独 task #118+ 决定 move 到 verdicts/ 或 gitignore) | ✅ track (但 location 不规范) |
| 是否 gitignore .DS_Store | ❌ (已 GitHub template 覆盖) | ✅ 重复添加 |
| 是否 gitignore __pycache__/ | ❌ (已 GitHub template 覆盖) | ✅ 重复添加 |

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| .gitignore 长度 | `wc -l .gitignore` | ✅ 197 → 222 (+25) |
| check-ignore data/ | `git check-ignore data/amazon_data/toys/training/file.tfrecord` | ✅ ignored |
| check-ignore scripts/pids/ | `git check-ignore scripts/pids/task123_pid` | ✅ ignored |
| check-ignore results/ | `git check-ignore results/` | ✅ ignored |
| tracked files 数 | `git ls-files \| wc -l` | ✅ 886 (+2 CLAUDE.md + requirements.txt) |
| untracked files 数 | `git ls-files --others --exclude-standard \| wc -l` | ✅ 162 (-92% from 2097) |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |

## 6. 关联

- 前置: Task #115/#116 (tracked verdicts/descriptions/scripts/papers)
- 后置: Task #118+ (top-level task109/119/123/27 _*.json/csv/png 处理, orphan src/ cleanup)

---

result: Task #117 — project-specific .gitignore 闭环. .gitignore append 11 project-specific rules (data/, logs/, products/, task_artifacts/, scripts/pids/, papers/notes/, papers/logs/, papers/grid_paper.txt, results/, log_tensorboard/, .claude/, .project-root). Track CLAUDE.md + requirements.txt (essential). Untracked noise 2097 → 162 (-92%). Tracked 884 → 886. dispatcher 5/5 PASS 不破坏.