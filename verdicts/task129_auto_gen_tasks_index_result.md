# Task #129 — Auto-Gen TASKS_INDEX (Break Drift Cycle)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: `scripts/task128_sync_task_docs.py` 实现 (~95 行). 自动从 `verdicts/` 扫 task<N>_*.md, 生成 TASKS_INDEX.md Recent closed tasks 段. 默认 `--from-id 101` (paper-defense 起点). 双模式: `--check` (exit 0 iff 同步, dispatcher 用) + default (auto diff + write). dispatcher 6/7 → 7/7 PASS. **drift cycle 终止**: future task 闭环时跑 `python3 scripts/task128_sync_task_docs.py` 即可同步.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| `scripts/task128_sync_task_docs.py` 创建 + py_compile | `python3 -m py_compile scripts/task128_sync_task_docs.py` | ✅ |
| default mode 生成 TASKS_INDEX.md | `python3 scripts/task128_sync_task_docs.py` | ✅ 31 tasks (#101 - #134) |
| --check mode 同步 | `python3 scripts/task128_sync_task_docs.py --check` | ✅ exit 0 |
| 修复 bug: extract_verdict_filename 没 sort → 跟 extract_subject 不一致 | 4 个 duplicates (#116/#118/#120/#125) 现在正确 | ✅ |
| 修复 bug: H1 regex 不允许 "Task #N result — subject" 格式 | 改成 `\b.*?[—\-]` 允许任意字符 | ✅ |
| dispatcher 7/7 PASS | `python3 scripts/all_audits.py` | ✅ |

---

## 2. 实现要点

### 2.1 _candidate_verdicts() — 单点排序

```python
def _candidate_verdicts(task_id: int) -> list[Path]:
    candidates = list(VERDICTS.glob(f"task{task_id}_*_result.md"))
    if not candidates:
        candidates = list(VERDICTS.glob(f"task{task_id}_*.md"))
    candidates.sort(key=lambda p: (not p.name.endswith("_result.md"), p.name))
    return candidates
```

两个 caller (extract_subject + extract_verdict_filename) 共享同一 sort, 确保选同一文件.

### 2.2 H1 regex — 兼容两种格式

- `# Task #N — <subject>` (标准)
- `# Task #N result — <subject>` (#116 类型)

regex: `r"^#\s*Task\s*#{task_id}\b.*?[—\-]\s*(.+)$"` — 允许 "Task #N" 和 "—" 之间任意字符.

### 2.3 --from-id 101 (paper-defense 起点)

不用 "last N" (语义错), 用 "from #101" (跟当前 TASKS_INDEX 显示范围匹配).

### 2.4 dispatcher 集成

`scripts/all_audits.py` AUDITS 加 task129 (check mode):
```python
("task129", "python3 scripts/task128_sync_task_docs.py --check"),
```

CI workflow `.github/workflows/audits.yml` 不需要改 — Task #125 dispatcher pattern 自动覆盖.

---

## 3. Before / After

### TASKS_INDEX.md Recent closed tasks section (after)

```
## Recent closed tasks (Task #101 - Task #134)

| Task | Subject | Status | Verdict |
|------|---------|--------|---------|
| #101 | REPRODUCE.md 复现包 + 环境验证脚本 闭环 | ✅ | verdicts/task101_reproduce_md_result.md |
...
| #116 | HG-Rec per-layer δ_95/diameter (原版 vs 改造版) | ✅ | verdicts/task116_hgrec_delta_per_layer_result.md |
| #117 | Project-Specific .gitignore (Verdict) | ✅ | verdicts/task117_gitignore_project_specific_result.md |
| #118 | HG-Rec codebook 利用率 / 碰撞率 (6 curvature 网格) | ✅ | verdicts/task118_hgrec_codebook_utilization_result.md |
...
| #128 | Housekeeping Cycle Closure (R8 §16 Cleanup + Drift Pattern Resolution) | ✅ | verdicts/task128_housekeeping_cycle_closure_result.md |
| #131 | phonism 4 seed 量化前 kNN vs Recall 相关性分析 verdict | ✅ | verdicts/task131_pre_quant_knn_vs_recall_result.md |
| #132 | 验证 Task #131 pre-quant R@5=0.00107 是否合理 | ✅ | verdicts/task132_verify_pre_quant_baselines_result.md |
| #134 | CLAUDE.md R9-Enforce + audit 脚本 verdict | ✅ | verdicts/task134_r9_enforce_audit_result.md |
```

注: #129, #130, #133 当前无 verdict file (在途), 自动跳过. Future 闭环后再 sync 自动加.

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否实现 auto-gen | ✅ 实现 (Task #128 spec, additive value) | ❌ 不实现 (drift cycle 继续) |
| audit name | ✅ task129 (跟 task ID 对齐, dispatcher order) | ❌ task128 (但 task128 已用作 sync script name) |
| script name vs task ID | ✅ task128_sync_task_docs.py (Task #128 spec 命名) | ❌ task129_*.py (spec 已固定) |
| 处理 duplicates (#116/#120/#125) | ✅ 字母序第一 (extract 共享 _candidate_verdicts) | ❌ 跳过 (loss of history) |
| TASKS_INDEX 重新生成模式 | ✅ auto-write + --check 双模式 | ❌ 只 write (CI 无法 check) |
| --from-id default | ✅ 101 (paper-defense 起点, 跟现有 TASKS_INDEX 范围对齐) | ❌ last N (语义错) |

---

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| sync script compiles | `python3 -m py_compile scripts/task128_sync_task_docs.py` | ✅ |
| default mode 生成 | `python3 scripts/task128_sync_task_docs.py` | ✅ 31 tasks |
| --check mode 同步 | `python3 scripts/task128_sync_task_docs.py --check` | ✅ exit 0 |
| extract_subject + extract_verdict_filename 一致 | manual check #116/#118/#120/#125 | ✅ |
| dispatcher 7/7 PASS | `python3 scripts/all_audits.py` | ✅ 7/7 |

---

## 6. Drift Cycle 终止

Before (Tasks #122 → #128):
- 每个 task 闭环时 TASKS_INDEX 漏 1 个 entry
- 下一个 task 必须做 "drift v(N+1)" 才能补齐
- self-perpetuating: 7 轮 housekeeping

After (Task #129):
- TASKS_INDEX Recent closed tasks 自动从 verdicts/ 生成
- future task 闭环后跑 `python3 scripts/task128_sync_task_docs.py` 即可同步
- dispatcher task129 --check 模式 CI 强制 review
- 不再需要 v2/v3/v4 drift fix

---

## 7. 关联

- 前置: Task #128 (housekeeping cycle closure + drift spec)
- 后置: drift cycle 终止. future task 闭环后跑 sync script 自动同步 TASKS_INDEX

---

result: Task #129 — auto-gen TASKS_INDEX 闭环. `scripts/task128_sync_task_docs.py` 实现 (~95 行, 双模式 auto-write + --check). dispatcher 6/7 → 7/7 PASS. drift cycle 终止. 4 个 duplicates (#116/#118/#120/#125) 通过共享 `_candidate_verdicts()` 排序一致. future task 闭环后跑 sync 即可同步 TASKS_INDEX, 不再需要手动 v2/v3/v4 drift fix.
