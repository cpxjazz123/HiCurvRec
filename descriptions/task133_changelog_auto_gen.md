# Task #133 — CHANGELOG.md Auto-Gen + 8th Dispatcher Audit

> **任务目的**: 终结 CHANGELOG.md docs drift cycle (类比 Task #129 TASKS_INDEX auto-gen). CHANGELOG.md [Unreleased] 段当前显示到 Task #127, #128-#132 全部缺失 (5 tasks drift). 复用 `_candidate_verdicts()` + H1 regex pattern, 创建 sync script append "### Recent housekeeping (auto-generated)" 子段. 整合到 dispatcher 作为第 8 个 audit (类比 Task #129).
> **执行日期**: 2026-07-24
> **状态**: 🟡 待启动

---

## 1. 背景

R11.3 自主决策 (per memory `drift-cycle-pattern-recognition`):
- **触发**: CHANGELOG.md [Unreleased] 段 grep `Task #1[12][0-9]` 命中 = 0 (Task #120+ 完全缺失), 显示截止到 #127 (Notes 段最后一段)
- **drift cycle 已启动**: 跟 Task #122-#128 TASKS_INDEX drift 一模一样模式 — 每个 task 闭环不强制更新 CHANGELOG → 下一个 task 必须 "drift v(N+1)" 才能补齐
- **终止方案**: 立即转 auto-gen (不做手动 housekeeping v2/v3), 类比 Task #129 TASKS_INDEX auto-gen pattern

**前置结论**:
- Task #129 已验证 auto-gen pattern: `_candidate_verdicts()` 共享排序 + H1 regex `^#\s*Task\s*#{N}\b.*?[—\-]\s*(.+)$` + 双模式 (default write / `--check` exit code)
- dispatcher pattern (Task #110/125) 已演进 4 → 7 audits, 添加 audit 只更新 `AUDITS` 列表, CI workflow 零修改
- Keep-a-Changelog 1.1.0 格式, CHANGELOG.md [Unreleased] 段是 current development 容器

---

## 2. 实验设计

**变量**: CHANGELOG.md [Unreleased] 段尾的 "Recent housekeeping (auto-generated)" 子段 (新增)
**保持不变**:
- CHANGELOG.md 已有 Added/Changed/Verified/Notes 4 子段 (Task #121 手维护内容, 不覆盖)
- CHANGELOG.md [1.0.0] 段及更早段 (静态)
- TASKS_INDEX.md (独立同步, Task #129 sync script 负责)

**新脚本设计**:
- `scripts/task133_sync_changelog.py` (~100 行)
- 复用 `_candidate_verdicts()` + H1 regex pattern (从 task128 import 或 inline copy)
- 扫 `verdicts/task<N>_*.md` (sorted by N, from-id >= 121 = post-submission housekeeping 起点)
- 生成 markdown table:
  ```markdown
  ### Recent housekeeping (auto-generated)

  | Task | Subject | Verdict |
  |------|---------|---------|
  | #128 | ... | `verdicts/...` |
  ```
- append 到 `[Unreleased]` 段尾 (不改其他段)
- 双模式: `default` (auto-write) / `--check` (exit 0 iff 同步)

**整合 dispatcher** (类比 Task #129):
- `scripts/all_audits.py` AUDITS 列表追加:
  ```python
  ("task133", "python3 scripts/task133_sync_changelog.py --check"),
  ```
- dispatcher 7/7 → 8/8 PASS

---

## 3. 决策触发

| 指标条件 | 结果 | 决策 |
|----------|------|------|
| CHANGELOG.md [Unreleased] 段尾 append "Recent housekeeping" 子段, 含 ≥ 5 个 task 行 (#128-#132) | ✅ drift cycle 终结 | 闭环 Task #133 |
| dispatcher 8/8 PASS | ✅ 8th audit 整合成功 | 闭环 Task #133 |
| sync script 兼容 Task #129 H1 regex (含 "Task #N result — subject") | ✅ 复用 pattern 验证 | 闭环 Task #133 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 sync script (~100 行) | ~10 min |
| 写 description + verdict | ~10 min |
| 整合 dispatcher 8th audit | ~3 min |
| 跑 dispatcher 8/8 PASS 验证 | ~30 sec |
| 总计 | ~25 min |

---

## 5. 风险与缓解

**风险 1**: CHANGELOG.md 已有 [Unreleased] 段 4 子段, append 时破坏原结构 → **缓解**: 只 append "### Recent housekeeping" 子段在末尾, 不动其他段
**风险 2**: Task #129 H1 regex 不兼容某些新 task H1 格式 → **缓解**: 复用 Task #129 已验证 regex, 含 `Task #N result — subject` 兼容
**风险 3**: dispatcher 8th audit 引入 race condition (其他 audit 修改 CHANGELOG 时冲突) → **缓解**: CHANGELOG 是 writeup only, dispatcher 串行执行无并发问题

---

## 6. 完成度跟踪

- [ ] 写 `scripts/task133_sync_changelog.py`
- [ ] 跑 `python3 -m py_compile scripts/task133_sync_changelog.py` (Rule 10 验证)
- [ ] 跑 `python3 scripts/task133_sync_changelog.py` (auto-write, append 子段)
- [ ] 整合 dispatcher AUDITS 列表 (8th audit)
- [ ] 跑 `python3 scripts/all_audits.py` 验证 8/8 PASS
- [ ] 跑 `python3 scripts/task128_sync_task_docs.py` 同步 TASKS_INDEX (auto-gen Task #133 行)
- [ ] 写 verdict `verdicts/task133_changelog_auto_gen_result.md`
- [ ] R8 §16 强制清理 (Task #133 闭环后从 §16 删除)