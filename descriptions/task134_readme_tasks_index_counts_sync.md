# Task #134 — README.md + TASKS_INDEX.md Counts Auto-Gen + 9th Dispatcher Audit

> **任务目的**: 终结 README.md + TASKS_INDEX.md 头文件 count 数字 drift (第 3 个 docs drift, 继 Task #129 TASKS_INDEX auto-gen + Task #133 CHANGELOG auto-gen 之后). README.md line 58 "125 entries" → 实际 133 unique task IDs (drift -8). TASKS_INDEX.md line 8-9 "132 numbered files covering 130 unique task IDs" → 实际 133+133 (drift -1, -3). TASKS_INDEX.md line 13 "261 files" → 实际 265 (drift -4). TASKS_INDEX.md line 3 Snapshot date 2026-07-24 post-Task #130 → 实际 post-Task #133. 复用 file count + regex pattern, 创建 sync script refresh 数字. 整合 dispatcher 作为第 9 个 audit.
> **执行日期**: 2026-07-24
> **状态**: 🟡 待启动

---

## 1. 背景

R11.3 自主决策 (per memory `drift-cycle-pattern-recognition`):
- **触发**: 2026-07-24 monitor tick grep README.md "125 entries" + TASKS_INDEX.md "132 numbered" 全部 drift
- **第 3 次发现 docs drift**: 跟 Task #122-#128 TASKS_INDEX drift + Task #133 CHANGELOG drift 一模一样模式 — 每次新 task 闭环不强制 refresh 数字 → 数字 stale
- **drift 量化**:
  - README.md line 58 "125 entries, R9 continuous" → 实际 133 unique task IDs (drift -8)
  - README.md line 61 "(~483 files)" → 实际 481 scripts (drift +2)
  - TASKS_INDEX.md line 8-9 "132 numbered files covering 130 unique task IDs (Task #1 - Task #130)" → 实际 133 numbered + 133 unique (drift -1 unique, -1 max)
  - TASKS_INDEX.md line 13 "261 files" → 实际 265 verdicts (drift -4)
  - TASKS_INDEX.md line 3 "Snapshot: 2026-07-24 (post Task #130)" → 实际 post-Task #133 (drift -3 tasks)
- **终止方案**: 立即转 auto-gen (不做手动数字 refresh v2/v3), 类比 Task #129 TASKS_INDEX auto-gen + Task #133 CHANGELOG auto-gen

**前置结论**:
- Task #129 已验证 auto-gen pattern: `_candidate_verdicts()` 共享排序 + H1 regex + 双模式 (default write / `--check` exit code)
- Task #133 已验证 CHANGELOG append-only pattern (保留手维护 4 子段)
- dispatcher pattern (Task #110/125/127/129/133) 已演进 4 → 8 audits, 添加 audit 只更新 `AUDITS` 列表, CI workflow 零修改

**跟 Task #129 #133 差异**:
- 数字 refresh (不添加新内容, 只 replace 现有数字)
- 不用 glob verdicts/ (Task #129 是 enumerate task IDs)
- 不用 append (CHANGELOG pattern) — 是 replace 数字 inline
- 数字来源: `os.walk` + `len()` 直接计算

---

## 2. 实验设计

**变量**: README.md line 58/61 + TASKS_INDEX.md line 3/8-9/13 的具体数字
**保持不变**:
- README.md 其他行 (项目介绍, 仓库结构注释, shields.io badges, license)
- TASKS_INDEX.md Coverage matrix (手维护, 含分类语义)
- TASKS_INDEX.md Recent closed tasks 段 (Task #129 sync 负责)
- CHANGELOG.md [Unreleased] 段 (Task #133 sync 负责)

**新脚本设计**:
- `scripts/task134_sync_doc_counts.py` (~120 行)
- 函数 `compute_counts()` → dict:
  - `n_descriptions_total`: `ls descriptions/ | wc -l`
  - `n_descriptions_unique_task_ids`: `ls descriptions/ | grep -oE 'task[0-9]+' | grep -oE '[0-9]+' | sort -n | uniq | wc -l`
  - `n_verdicts_total`: `ls verdicts/ | wc -l`
  - `n_scripts_total`: `ls scripts/ | wc -l`
  - `max_task_id`: `ls descriptions/ | grep -oE 'task[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1`
  - `snapshot_task_id`: max_task_id (assume latest closed task == snapshot reference)
- 函数 `apply_to_readme(text, counts)` → str: replace line 58 + line 61 with actual numbers
- 函数 `apply_to_tasks_index(text, counts)` → str: replace line 3 snapshot + line 8-9 unique count + line 13 verdicts count
- 双模式: default auto-write / `--check` exit 0 iff 同步

**整合 dispatcher** (类比 Task #133):
- `scripts/all_audits.py` AUDITS 列表追加:
  ```python
  ("task134", "python3 scripts/task134_sync_doc_counts.py --check"),
  ```
- dispatcher 8/8 → 9/9 PASS

---

## 3. 决策触发

| 指标条件 | 结果 | 决策 |
|----------|------|------|
| README.md + TASKS_INDEX.md 数字 refresh 后, grep 验证 0 drift | ✅ drift 终结 | 闭环 Task #134 |
| dispatcher 9/9 PASS | ✅ 9th audit 整合成功 | 闭环 Task #134 |
| sync script 兼容 task27/task67 duplicates (2 files per ID) | ✅ unique IDs 而非 file count | 闭环 Task #134 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 sync script (~120 行) | ~15 min |
| 写 description + verdict | ~10 min |
| 整合 dispatcher 9th audit | ~3 min |
| 跑 dispatcher 9/9 PASS 验证 | ~30 sec |
| 总计 | ~30 min |

---

## 5. 风险与缓解

**风险 1**: TASKS_INDEX.md line 8-9 含分类 comment ("132 numbered files covering 130 unique task IDs"), 不能直接 replace 数字 → **缓解**: 用 regex `\bNNN\b` 替换 actual 数字, 保留前后文
**风险 2**: task27/task67 duplicates (2 files per ID) → unique ID count 不等于 file count → **缓解**: script 用 unique IDs 数字 (133) 而非 file count (137), 跟 R9 contiguous 语义一致
**风险 3**: Snapshot date 误判 (例如 max_task_id 不等于 snapshot reference task) → **缓解**: script 直接用 max_task_id 作为 snapshot reference, reviewer 一致理解

---

## 6. 完成度跟踪

- [ ] 写 `scripts/task134_sync_doc_counts.py`
- [ ] 跑 `python3 -m py_compile scripts/task134_sync_doc_counts.py` (Rule 10 验证)
- [ ] 跑 `python3 scripts/task134_sync_doc_counts.py` (auto-write, refresh 数字)
- [ ] 整合 dispatcher AUDITS 列表 (9th audit)
- [ ] 跑 `python3 scripts/all_audits.py` 验证 9/9 PASS
- [ ] 跑 `python3 scripts/task128_sync_task_docs.py` 同步 TASKS_INDEX (auto-gen Task #134 行)
- [ ] 跑 `python3 scripts/task133_sync_changelog.py` 同步 CHANGELOG (auto-gen Task #134 行)
- [ ] 写 verdict `verdicts/task134_readme_tasks_index_counts_sync_result.md`
- [ ] R8 §16 强制清理 (Task #134 闭环后从 §16 删除)