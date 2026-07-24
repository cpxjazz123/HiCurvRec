# Task #132 — Save Infrastructure Pattern Learnings to Memory

> **任务目的**: 131 个 task 闭环产生 3 个 reusable infrastructure patterns 没保存到 memory: (a) TASKS_INDEX auto-gen (Task #129) — drift cycle 终结方案; (b) Drift cycle pattern recognition (Task #128) — housekeeping self-perpetuating 早期信号; (c) Dispatcher pattern (Task #110/125/127/129) — single audit entry point. 保存为 3 个 memory entries, future projects 可复用.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

现有 11 memory entries 都是 specific task findings (e.g. Task #70 Ollivier 曲率, Task #87 Table 2 ranking). 没有 infrastructure / process pattern memory. 131 个 task 闭环产生 3 个 reusable 经验值得 memorialize:

1. **TASKS_INDEX auto-gen pattern** (Task #129): `scripts/task128_sync_task_docs.py` 双模式 (auto-write + --check). 终结 drift cycle. 通用 pattern: 手维护文档 → 自动从 source-of-truth 生成.
2. **Drift cycle pattern** (Task #128): housekeeping self-perpetuating drift (每个 task 创建 docs drift → 需 v2/v3 fix → 又创建新 drift). 早期信号识别 + 终结方案.
3. **Dispatcher pattern** (Task #110/125/127/129): single audit entry point `scripts/all_audits.py` 跑 N audit + 返回 unified exit code. CI workflow 不需要改. 添加 audit 只更新 AUDITS 列表.

**Task #132 = save 3 memory entries**

---

## 2. 实验设计 (writeup only)

### 2.1 Memory entry #1: TASKS_INDEX auto-gen pattern

File: `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/tasks-index-auto-gen-pattern.md`

```markdown
---
name: tasks-index-auto-gen-pattern
description: TASKS_INDEX auto-gen pattern — drift cycle 终结方案 (Task #129). 手维护 docs → 自动从 source-of-truth 生成 + --check mode dispatcher integration.
metadata:
  type: reference
---

# TASKS_INDEX Auto-Gen Pattern

手维护 TASKS_INDEX (含 Recent closed tasks table) 会产生 self-perpetuating drift: 每个 task 闭环时 TASKS_INDEX 漏 1 个 entry, 下一个 task 必须做 "drift v(N+1)" 才能补齐.

**终结方案** (Task #129): auto-gen script `scripts/task128_sync_task_docs.py`.

设计:
- 扫 `verdicts/task<N>_*.md` (sorted by N) → 提取每个 H1 的 subject
- 生成 Recent closed tasks markdown table (含 Subject + Verdict 列)
- 替换 TASKS_INDEX.md "## Recent closed tasks" 段 (保留所有其他 hand-curated 段)
- 双模式:
  - `default` (auto diff + write)
  - `--check` (exit 0 iff 同步, dispatcher 7th audit 用)

集成到 dispatcher:
```python
("task129", "python3 scripts/task128_sync_task_docs.py --check"),
```

**Why**: CI workflow 不需要改 (dispatcher pattern). 添加 audit 只更新 AUDITS 列表. future task 闭环后跑 sync 即可同步 TASKS_INDEX.

**How to apply**: 任何手维护的 docs (含动态 generated 段) 都可迁移此 pattern. 关键是 source-of-truth (verdicts/) 是 immutable, derived docs (TASKS_INDEX) 是 auto-generated.

**Related**: [[drift-cycle-pattern-recognition]], [[dispatcher-pattern]]
```

### 2.2 Memory entry #2: Drift cycle pattern

File: `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/drift-cycle-pattern-recognition.md`

```markdown
---
name: drift-cycle-pattern-recognition
description: Housekeeping drift cycle pattern — self-perpetuating (Task #128). 每个 housekeeping task 创建新 docs drift, 需 v2/v3 fix. 早期信号识别 + auto-gen 终结.
metadata:
  type: feedback
---

# Drift Cycle Pattern Recognition

观察: housekeeping task chain (Task #122 → #128, 7 轮) 显示 self-perpetuating drift pattern:

| Task | 修的 drift | 创造的 drift |
|------|----------|----------|
| #122 | TASKS_INDEX 漏 #113-#121 | (无) |
| #123 | REPRODUCE.md script 7 处不存在 | (无) |
| #124 | README/TASKS_INDEX 漏 #115-#123 | (无) |
| #125 | CI workflow 4 separate → 1 dispatcher | (无) |
| #126 | TASKS_INDEX/CHANGELOG 漏 #124-#125 | 漏 #126 自己 |
| #127 | TASKS_INDEX/CHANGELOG 漏 #126 | 漏 #127 自己 |
| #128 | TASKS_INDEX/CHANGELOG 漏 #127 | (R8 强制清理后 stop) |

**早期信号**:
1. 每个 task 闭环时 TASKS_INDEX/CHANGELOG 漏 1 个 entry
2. 下一个 task 必须做 "drift v(N+1)" 才能补齐
3. 7 轮后仍未终止 (Task #126 漏 #126 → #127 漏 #127)

**根因**: TASKS_INDEX/CHANGELOG 是手维护文档, 闭环时不强制同步.

**终结方案** (Task #129): auto-gen script + dispatcher --check audit. 详见 [[tasks-index-auto-gen-pattern]].

**Why**: 识别 drift cycle 后立即停止手动 housekeeping, 转 auto-gen. 否则边际价值 < 0.

**How to apply**: 任何手维护 docs 项目, 第 2 次发现 "drift v(N+1)" task 时立即转 auto-gen.

**Related**: [[tasks-index-auto-gen-pattern]]
```

### 2.3 Memory entry #3: Dispatcher pattern

File: `/home/wlia0047/.claude/projects/-fs04-ar57-wenyu-GeneRec/memory/dispatcher-pattern.md`

```markdown
---
name: dispatcher-pattern
description: Single audit dispatcher pattern — N audit scripts → 1 entry point (Task #110/125). CI workflow 不需要改, 添加 audit 只更新 AUDITS 列表.
metadata:
  type: reference
---

# Dispatcher Pattern (Single Audit Entry Point)

**问题**: N audit scripts 在 CI workflow 里 N separate steps. 添加新 audit 需要改 workflow YAML (CI workflow drift). Reviewer 跑 audit 需要记 N 个命令.

**解决方案** (Task #110/125): single dispatcher `scripts/all_audits.py` 跑 N audit + 返回 unified exit code.

设计:
```python
AUDITS: list[tuple[str, str]] = [
    ("task101", "python3 scripts/task101_verify_env.py ..."),
    ("task103", "python3 scripts/task103_paper_claims_audit.py"),
    # ... (添加 audit 只更新这里)
]
# run each via subprocess.run, return 0 iff all pass
```

Reviewer-facing single entry point: `python3 scripts/all_audits.py`. 返回 0 iff all audits pass.

CI workflow `.github/workflows/audits.yml` 简化:
```yaml
- name: Single audit dispatcher
  run: python3 scripts/all_audits.py
```

**Why**:
- Reviewer 只记 1 个命令
- CI workflow 不需要改 (添加 audit 只更新 dispatcher AUDITS 列表)
- 统一 exit code (CI success/fail 二元信号)
- artifact upload + summary step 自动覆盖新 audit (per dispatcher output)

**How to apply**: 任何项目有 N 个 verification/audit/lint 脚本. 创建 dispatcher 统一管理. 4+ audit 时开始有价值.

**演进** (Task #125/127/129): dispatcher 从 4 audit → 7 audit, CI workflow 零修改.

**Related**: [[tasks-index-auto-gen-pattern]]
```

### 2.4 更新 MEMORY.md 索引

Append 3 行 to MEMORY.md:
```
- [TASKS_INDEX auto-gen pattern](tasks-index-auto-gen-pattern.md) — Task #129: drift cycle 终结方案. 通用 pattern: 手维护 docs → 自动从 source-of-truth 生成 + --check mode.
- [Drift cycle pattern](drift-cycle-pattern-recognition.md) — Task #128: housekeeping self-perpetuating drift 早期信号 + auto-gen 终结方案.
- [Dispatcher pattern](dispatcher-pattern.md) — Task #110/125: N audit scripts → 1 entry point. CI workflow 不需要改, 添加 audit 只更新 AUDITS 列表.
```

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| 3 个 memory entries 创建 | ✅ 闭环 |
| MEMORY.md 索引更新 (3 行) | ✅ 闭环 |
| dispatcher 7/7 PASS 不破坏 | ✅ 闭环 (memory 不影响 dispatcher) |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 保存几个 memory entries | ✅ 3 个 (TASKS_INDEX auto-gen + drift cycle + dispatcher) | ❌ 1 个 (太分散) 或 ❌ 5+ 个 (over-save) |
| memory type | ✅ reference (TASKS_INDEX auto-gen + dispatcher) / feedback (drift cycle) | ❌ project (跟 task-specific entries 重复) |
| MEMORY.md 索引更新 | ✅ Append 3 行 (link 到新 entries) | ❌ 不更新 (但下次 recall 找不到) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 3 个 memory .md files | ~5 min |
| 更新 MEMORY.md 索引 | ~1 min |
| git commit + §16 | ~2 min |
| **总计** | **~8 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: memory 写错, future recall 用错信息
  → **缓解**: 严格按现有 memory entry 格式 (frontmatter + body + Related links)

**风险 2**: memory 跟 task-specific entries 重复
  → **缓解**: type 选 `reference`/`feedback` (process pattern), 不是 `project` (specific task finding)

---

## 7. 完成度跟踪

- [x] 写 descriptions/task132_memory_infrastructure_patterns.md (本文件)
- [ ] 写 memory/tasks-index-auto-gen-pattern.md
- [ ] 写 memory/drift-cycle-pattern-recognition.md
- [ ] 写 memory/dispatcher-pattern.md
- [ ] 更新 MEMORY.md 索引 (3 行 append)
- [ ] git commit + §16 update

---

## 8. 关联

- 前置: Task #129 (auto-gen TASKS_INDEX) + #128 (drift pattern) + #110/125 (dispatcher)
- 后置: future Claude sessions 在 similar 项目场景可 recall 这 3 个 patterns

---

**核心交付**: 3 个 infrastructure memory entries (TASKS_INDEX auto-gen + drift cycle + dispatcher). 通用 patterns, future projects 可复用. 0 GPU ~8 min.
