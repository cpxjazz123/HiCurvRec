# Task #128 — Housekeeping Cycle Closure (R8 §16 Cleanup + Drift Pattern Resolution)

> **任务目的**: (1) §16 R8 cleanup: 表格主体清空, 只留当前活跃任务 (空); header 文本更新 "Task #125 闭环" → "Task #127 闭环 + housekeeping cycle closure"; 移除 §16 表格里所有已完成任务的 R8 强制清理行 (R8 说必须直接删除, 不是写状态描述); (2) Drift pattern 闭环: 写 verdict 明确说明 "housekeeping drift 是 self-perpetuating (每个 task 创建 docs drift → 需 v2/v3 fix → 又创建新 drift)", 推荐方案 = `scripts/task128_sync_task_docs.py` 自动生成 TASKS_INDEX Recent closed tasks 段 (消除手动维护漂移).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #125 → #126 → #127 三轮 housekeeping 显示一个 self-perpetuating drift pattern:

| Task | 修的 drift | 创造的 drift |
|------|----------|----------|
| #125 | CI workflow 4 separate → 1 dispatcher | (无新) |
| #126 | TASKS_INDEX/CHANGELOG 漏 #124-#125 | 漏 #126 自己 |
| #127 | TASKS_INDEX/CHANGELOG 漏 #126 | 漏 #127 自己 |

每个 task 闭环时 TASKS_INDEX/CHANGELOG 漏 1 个 entry, 下一个 task 必须做 "drift v(N+1)" 才能补齐. 这是因为 TASKS_INDEX/CHANGELOG 是手维护的, 闭环时不强制同步.

**Task #128 = housekeeping cycle closure**, R8 §16 强制清理 (移除已完成 R8 描述行), 推荐 auto-gen 解决方案跳出循环.

---

## 2. 实验设计 (writeup only)

### 2.1 §16 R8 强制清理

按 R8: "❌ 禁止在 §16 写'已归档'/'已暂停'/'后台监控'/'历史归档'等历史性条目行 (必须直接删除该任务描述行)". 

当前 §16 表格状态:
- 行 1: "**#126** ..." (✅ 已完成) → 移到 R8 历史段
- 行 2: "**(空 — R8 强制清理: Task #126 ...)**" → 删 (R8 说必须直接删除, 不是写历史段)
- 行 3: "**#127** ..." (✅ 已完成) → 移到 R8 历史段
- 行 4: "**(空 — R8 强制清理: Task #127 ...)**" → 删

新 §16 状态: 表格主体空白, header 显示当前状态.

### 2.2 §16 header 文本更新

当前: "🟢 0 个活跃 (2026-07-24 更新, Task #125 CI workflow dispatcher sync 闭环 ...)"

更新为: "🟢 0 个活跃 (2026-07-24 更新, Task #127 6th audit + drift v3 闭环 + housekeeping cycle closure. dispatcher 6/6 PASS. reviewer-facing docs 跟 git 状态 cross-validation 闭合到 #127. **Housekeeping cycle 暂停**: drift pattern 自我强化, 推荐 auto-gen 解决方案跳出循环)"

### 2.3 推荐 auto-gen 解决方案 (R11.3)

写一个 `scripts/task128_sync_task_docs.py` (optional, future):
- 扫 `descriptions/task<N>_*.md` + `verdicts/task<N>_*.md`
- 自动生成 TASKS_INDEX.md "Recent closed tasks" 段 + "Coverage matrix" 段
- 一行命令: `python3 scripts/task128_sync_task_docs.py` 同步

**Task #128 = 推荐 + 写出 spec**, 不实际实现 (over-engineering for current state). Future task (#129+) 可实现.

### 2.4 不实现 6th+ audit, 不 gitignore src/configs/tools

- 6th audit 已加 (Task #127), 7th audit 无明显价值
- src/configs/tools 不 gitignore (REPRODUCE §0 onboarding 期望 visible)

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| §16 R8 强制清理 (删 R8 描述行 + 移已完成行) | ✅ 闭环 |
| §16 header 文本更新 | ✅ 闭环 |
| TASKS_INDEX/CHANGELOG 已同步 #127 | ✅ (Task #127 已闭环) |
| dispatcher 6/6 PASS 不破坏 | ✅ 闭环 |
| verdict 写 drift pattern + 推荐 auto-gen spec | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| §16 R8 cleanup 强度 | ✅ 强清 (删 R8 描述行) | ❌ 保留 (但 R8 明文禁止) |
| auto-gen TASKS_INDEX 是否实现 | ❌ 只 spec 不实现 (over-engineering for current state) | ✅ 实现 (但 Task #129+ 边界外) |
| housekeeping cycle 是否继续 | ❌ 暂停 (drift pattern 自我强化, marginal value < 0) | ✅ 继续 (但每次都制造新 drift) |
| §16 header 文本 | ✅ 改 Task #127 closure (新状态) | ❌ 保留 Task #125 (过时) |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| §16 表格 R8 cleanup | ~3 min |
| §16 header 文本更新 | ~1 min |
| 写 verdict (drift pattern 分析) | ~3 min |
| git commit + §16 | ~2 min |
| **总计** | **~10 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: §16 R8 cleanup 把已完成 task 行删了, 后续需要回看历史需要 git log
  → **缓解**: verdicts/ + descriptions/ 文件保留, git log 可追溯; R8 明文说必须直接删

**风险 2**: auto-gen TASKS_INDEX 推荐方案不被采纳
  → **缓解**: Task #128 verdict 记录 spec, future task 可实现

**风险 3**: housekeeping cycle 暂停后, 后续 task 创建的 docs drift 无修复
  → **缓解**: dispatcher task114 已含 R9 contiguous check (description/ verdict ID pairing), 强制下次 review 触发 fix

---

## 7. 完成度跟踪

- [x] 写 descriptions/task128_housekeeping_cycle_closure.md (本文件)
- [ ] §16 表格 R8 cleanup (删 R8 描述行 + 移已完成行)
- [ ] §16 header 文本更新
- [ ] 写 verdict (drift pattern + auto-gen spec)
- [ ] git commit + §16 update
- [ ] dispatcher 6/6 PASS 验证

---

## 8. 关联

- 前置: Task #127 (6th audit + drift v3)
- 后置: 暂停 housekeeping cycle. Future task #129+ 若需 sync docs, 推荐实现 task128_sync_task_docs.py auto-gen 解决方案

---

**核心交付**: §16 R8 强制清理 (表格清空) + housekeeping cycle closure (drift pattern 总结 + auto-gen spec 推荐). dispatcher 6/6 PASS 不破坏. reviewer-facing docs 现在跟 git 状态最小化 cross-validation (TASKS_INDEX/CHANGELOG 只可能 lag 1 个 task).
