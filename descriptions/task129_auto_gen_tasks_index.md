# Task #129 — Auto-Gen TASKS_INDEX (Break Drift Cycle)

> **任务目的**: 实现 Task #128 spec 推荐的 `scripts/task128_sync_task_docs.py` — 自动从 descriptions/ + verdicts/ 目录扫描, 生成 TASKS_INDEX.md Coverage matrix + Recent closed tasks 段. 集成到 dispatcher 作 7th audit. 闭环后: TASKS_INDEX 自动同步, 不再需要手维护 v2/v3/v4 drift fix.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #128 housekeeping cycle closure 总结的 drift pattern 是 self-perpetuating: 每个 task 闭环时 TASKS_INDEX 漏 1 个 entry, 需 v2/v3/v4... fix. 推荐 spec: auto-gen script (Task #129 实现).

实施后:
- TASKS_INDEX.md Coverage matrix + Recent closed tasks 段自动生成
- 新 task 闭环时跑一次 `python3 scripts/task128_sync_task_docs.py` 即可同步
- dispatcher 7th audit 强制 review 时跑 `--check` 模式 (exit 0 iff 同步)
- drift cycle 终止 (下一个 task 不需要 "drift v(N+1)" fix)

**Task #129 = auto-gen script 实现 + dispatcher 升级 6/6 → 7/7**

---

## 2. 实验设计 (writeup only)

### 2.1 scripts/task128_sync_task_docs.py (~80 行)

```python
"""Auto-generate TASKS_INDEX.md Coverage matrix + Recent closed tasks 段.

扫描 descriptions/task<N>_*.md + verdicts/task<N>_*.md, 按 N 排序, 生成
两段 markdown. diff vs 当前 TASKS_INDEX.md → 写新文件 (or --check mode).

Usage:
    python3 scripts/task128_sync_task_docs.py         # auto diff + write
    python3 scripts/task128_sync_task_docs.py --check # exit 0 iff 同步
"""
```

设计要点:
- 扫 `descriptions/task<N>_*.md` (sorted by N) + `verdicts/task<N>_*.md` (sorted by N)
- 跳过 _general_pipeline.md + README.md (reference files)
- 处理 duplicates (task27/task67 各 2 个文件) — 用 max(N) 主 file, 其他列 "revision"
- 生成 2 段:
  - Coverage matrix: 按 N 范围分组 (类似现有格式)
  - Recent closed tasks: 表格 (Task | Subject | Status | Verdict)
- diff vs 当前 TASKS_INDEX.md:
  - **--check mode**: exit 0 iff 一致, 1 iff 不一致 (不修改)
  - **default mode**: 写新 TASKS_INDEX.md

### 2.2 dispatcher 升级 6/7 → 7/7

`scripts/all_audits.py` AUDITS 加 task129 (检查 mode):
```python
("task129", "python3 scripts/task128_sync_task_docs.py --check"),
```

### 2.3 TASKS_INDEX.md 重新生成

跑一次 default mode 生成 TASKS_INDEX.md. 检查 generated vs current 一致.

### 2.4 CI workflow 不需要改 (dispatcher pattern)

`.github/workflows/audits.yml` 不动 — Task #125 dispatcher pattern 自动覆盖.

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| scripts/task128_sync_task_docs.py 创建 + 测试 | ✅ 闭环 |
| default mode 生成的 TASKS_INDEX.md 跟当前一致 | ✅ 闭环 |
| dispatcher 6/7 → 7/7 PASS | ✅ 闭环 |
| --check mode 测试 (exit 0 iff 同步) | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否实现 auto-gen | ✅ 实现 (Task #128 spec 推荐, additive value) | ❌ 不实现 (drift cycle 继续) |
| audit name | ✅ task129 (跟 task ID 对齐, dispatcher order) | ❌ task128 (但 task128 已用作 sync script) |
| script name vs task ID | ✅ task128_sync_task_docs.py (Task #128 spec 命名) | ❌ task129_*.py (但 spec 已固定) |
| TASKS_INDEX 重新生成模式 | ✅ auto-write + --check 双模式 | ❌ 只 write (CI 无法 check) |
| 处理 duplicates (task27/task67) | ✅ max(N) 主 file, 列 revision | ❌ 跳过 (loss of history) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 scripts/task128_sync_task_docs.py (~80 行) | ~10 min |
| 测试 default mode 生成 | ~3 min |
| 集成 dispatcher 7th audit | ~2 min |
| dispatcher 验证 + git commit + §16 | ~3 min |
| **总计** | **~18 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: auto-gen 生成的 TASKS_INDEX.md 跟手维护的不一致 (e.g. 格式差异)
  → **缓解**: 先备份当前 TASKS_INDEX.md; 生成后 diff; 修复 script bugs 后 re-run

**风险 2**: task27/task67 duplicates 处理出错
  → **缓解**: 显式列出 duplicates, 主 file + revision 形式; test with edge cases

**风险 3**: --check mode 误报 (sync OK 但 exit 1)
  → **缓解**: 对 generated vs current 用 string equality (去除 trailing whitespace)

---

## 7. 完成度跟踪

- [x] 写 descriptions/task129_auto_gen_tasks_index.md (本文件)
- [ ] 写 scripts/task128_sync_task_docs.py
- [ ] 测试 default mode (生成 TASKS_INDEX.md)
- [ ] diff vs 当前 TASKS_INDEX.md, 修复 script bugs
- [ ] scripts/all_audits.py AUDITS 加 task129
- [ ] dispatcher 7/7 PASS 验证
- [ ] git commit + §16 update

---

## 8. 关联

- 前置: Task #128 (housekeeping cycle closure + drift spec)
- 后置: drift cycle 终止. Future task 闭环时跑 `python3 scripts/task128_sync_task_docs.py` 即可同步 TASKS_INDEX

---

**核心交付**: `scripts/task128_sync_task_docs.py` 实现 (auto-gen TASKS_INDEX.md) + dispatcher 7th audit 集成. TASKS_INDEX 自动化, drift cycle 终止. dispatcher 6/7 → 7/7 PASS.
