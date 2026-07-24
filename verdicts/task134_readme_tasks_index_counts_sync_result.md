# Task #134 — README.md + TASKS_INDEX.md Counts Auto-Gen + 9th Dispatcher Audit

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: (a) `scripts/task134_sync_doc_counts.py` (~150 行, 数字 refresh 而非内容 append); (b) README.md line 58 "125 entries" → "134 entries", line 61 "~483 files" → "~482 files"; (c) TASKS_INDEX.md line 3 snapshot "post Task #130" → "post Task #134", line 8 "134 files total — 132 numbered" → "138 files total — 136 numbered", line 9 "130 unique" → "134 unique", line 13 "261 files" → "265 files"; (d) `scripts/all_audits.py` 8 → 9 audit 整合. dispatcher 9/9 PASS. 终结第 3 个 docs drift cycle (README + TASKS_INDEX counts).

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| `scripts/task134_sync_doc_counts.py` 创建 | `ls scripts/task134_sync_doc_counts.py` | ✅ ~150 行 |
| Rule 10 py_compile 验证 | `python3 -m py_compile scripts/task134_sync_doc_counts.py` | ✅ |
| README.md 数字 refresh | `grep "134 entries" README.md` | ✅ |
| TASKS_INDEX.md 数字 refresh (4 处) | `sed -n '3p;8,9p;13p' TASKS_INDEX.md` | ✅ |
| `--check` exit 0 | `python3 scripts/task134_sync_doc_counts.py --check` | ✅ |
| dispatcher 9/9 PASS | `python3 scripts/all_audits.py` | ✅ |
| R9 层 2 (descriptions/ 1-134 无空洞) | `seq 1 134 diff ls descriptions` | ✅ |
| TASKS_INDEX sync 含 #134 | `grep "#134" TASKS_INDEX.md` | ✅ (via task128) |

---

## 2. 第 3 个 docs drift cycle 终结

**触发**: 2026-07-24 monitor tick grep README.md "125 entries" + TASKS_INDEX.md "132 numbered" → 全部 drift (实际 134 unique IDs).

**drift 量化** (5 处):
| 文件 | 行 | drift 前 | drift 后 | 实际值 |
|------|----|---------|---------|--------|
| README.md | 58 | 125 entries | 134 entries | 134 unique task IDs |
| README.md | 61 | ~483 files | ~482 files | 482 scripts |
| TASKS_INDEX.md | 3 | post Task #130 | post Task #134 | post Task #134 |
| TASKS_INDEX.md | 8 | 134 files total — 132 numbered | 138 files total — 136 numbered | 138 files / 136 numbered |
| TASKS_INDEX.md | 9 | 130 unique task IDs (Task #1 - Task #130) | 134 unique (Task #1 - Task #134) | 134 unique |
| TASKS_INDEX.md | 13 | 261 files (Task #1 - Task #129) | 265 files (Task #1 - Task #134) | 265 files |

**终止方案**: 类比 Task #129 + Task #133 auto-gen pattern. 但差异:
- Task #129: 整段覆盖 TASKS_INDEX.md "Recent closed tasks" 段
- Task #133: append CHANGELOG.md [Unreleased] 段尾 "Recent housekeeping" 子段
- **Task #134: 数字 refresh inline (prefix match regex, 保留 trailing text)**

**regex 设计** (跟 Task #129 #133 差异):
- 用 prefix match (如 `\(post Task #\d+`) 替换数字, **保留 trailing text** (如 " final state closure" / "contiguous per R9; ...")
- 这是 sync script 第一次写时漏匹配 (false positive — `--check` 报 "in sync" 但实际 2 处 stale)
- 修复: prefix match + suffix preservation, 单行 regex 用 `$` end-of-line

**compute_counts() 输出**:
```python
{
    "n_descriptions_total": 138,       # ls descriptions/ | wc -l
    "n_descriptions_numbered": 136,   # task<N>_*.md only (excludes 2 reference)
    "n_descriptions_unique_task_ids": 134,  # 134 unique (含 #27 #67 duplicates)
    "max_task_id": 134,
    "n_verdicts_total": 265,           # ls verdicts/ | wc -l (含 3 orphans)
    "n_scripts_total": 482,           # ls scripts/ | wc -l
}
```

---

## 3. 9th dispatcher audit 整合

`scripts/all_audits.py`:
```python
AUDITS: list[tuple[str, str]] = [
    ...
    ("task129", "python3 scripts/task128_sync_task_docs.py --check"),  # TASKS_INDEX table
    ("task133", "python3 scripts/task133_sync_changelog.py --check"),  # CHANGELOG subsection
    ("task134", "python3 scripts/task134_sync_doc_counts.py --check"), # README + TASKS_INDEX counts (NEW)
]
```

Header 更新: 8 → 9 paper defense audit scripts.

**dispatcher 演进**: 4 → 9 audits 跨 6 task (Task #110/125/127/129/133/134):
- Task #110: 4 audits (task101/103/105/106)
- Task #114: 5 audits (加 verdict integrity)
- Task #125: CI workflow 改 dispatcher pattern
- Task #127: 6 audits (加 script syntax)
- Task #129: 7 audits (加 TASKS_INDEX sync)
- Task #133: 8 audits (加 CHANGELOG sync)
- **Task #134: 9 audits (加 README + TASKS_INDEX counts sync)**

每次添加 audit 零 CI workflow 修改.

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 终结 vs 手动数字 refresh | ✅ auto-gen (per memory `drift-cycle-pattern-recognition`) | ❌ 手动 refresh 数字 (drift v2/v3/v4, 边际价值 < 0) |
| prefix match vs full line replace | ✅ prefix match (保留 trailing text "final state closure" / "contiguous per R9;...") | ❌ full line replace (破坏 review-friendly 注释) |
| sync 范围 (单 doc vs 多 doc) | ✅ multi-doc sync (README + TASKS_INDEX 共 6 处数字) | ❌ 单 doc sync (重复 dispatcher audit) |
| unique IDs vs file count | ✅ unique IDs (R9 contiguous 语义) | ❌ file count (134 ≠ 138, 跟 R9 不一致) |
| snapshot date 来源 | ✅ max_task_id (latest closed task == snapshot reference) | ❌ hard-coded (drift 风险) |

---

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| sync script 语法 | `python3 -m py_compile` | ✅ PASS |
| sync auto-write | `python3 scripts/task134_sync_doc_counts.py` | ✅ 6 numbers refreshed |
| sync --check (修复 regex 后) | `python3 scripts/task134_sync_doc_counts.py --check` | ✅ in sync |
| dispatcher 语法 | `python3 -m py_compile scripts/all_audits.py` | ✅ PASS |
| dispatcher 9/9 | `python3 scripts/all_audits.py` | ✅ |
| R9 contiguous 1-134 | `seq 1 134 diff ls descriptions` | ✅ no gaps |
| TASKS_INDEX #134 行 | `grep "#134" TASKS_INDEX.md` | ✅ (via task128 sync) |
| README.md 134 entries | `grep "134 entries" README.md` | ✅ |
| TASKS_INDEX 4 numbers | `sed -n '3p;8,9p;13p' TASKS_INDEX.md` | ✅ all 4 refreshed |

---

## 6. 关联

- **前置**: Task #129 (TASKS_INDEX auto-gen pattern), Task #133 (CHANGELOG auto-gen pattern), Task #132 (memory `drift-cycle-pattern-recognition` saved)
- **后续**: future Claude sessions 闭环 task 后跑 3 个 sync 即可同步所有手维护 docs:
  ```bash
  python3 scripts/task128_sync_task_docs.py \
    && python3 scripts/task133_sync_changelog.py \
    && python3 scripts/task134_sync_doc_counts.py
  ```
- **平行**: Task #129 (TASKS_INDEX table) + Task #133 (CHANGELOG subsection) + Task #134 (README + TASKS_INDEX counts) 形成 docs sync 完整覆盖

---

result: Task #134 — README.md + TASKS_INDEX.md counts auto-gen + 9th dispatcher audit 闭环. `scripts/task134_sync_doc_counts.py` (~150 行) 创建并 py_compile PASS. 6 numbers refreshed (README 2 + TASKS_INDEX 4). 终结第 3 个 docs drift cycle (类比 Task #129 #133 auto-gen pattern). dispatcher 8/8 → 9/9 PASS. R9 contiguous 1-134. future task 闭环后跑 3 sync 即可同步所有 docs.