# Task #128 — Housekeeping Cycle Closure (R8 §16 Cleanup + Drift Pattern Resolution)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: (a) §16 R8 强制清理: 删除 #126/#127 已完成活跃行 + 删除 R8 描述行 (R8 明文禁止"已归档/已暂停/后台监控/历史归档"等历史条目), 表格主体空白; (b) §16 header 文本更新 "Task #125 闭环" → "Task #127 + #128 housekeeping cycle closure. dispatcher 6/6 PASS. Housekeeping cycle 暂停"; (c) Drift pattern 总结 + 推荐 auto-gen 解决方案 spec. dispatcher 6/6 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| §16 删除 #126/#127 活跃行 | `grep -c "^\| \*\*#12[67]\*\*" loop.md` | ✅ 0 |
| §16 删除 R8 描述行 | `grep -c "R8 强制清理: Task #12[67]" loop.md` | ✅ 0 |
| §16 表格主体空白 | `grep -cE "^\| \(空 \|" loop.md` | ✅ 1 |
| §16 header 含 housekeeping cycle closure 描述 | `grep "housekeeping cycle closure" loop.md` | ✅ |
| dispatcher 6/6 PASS | `python3 scripts/all_audits.py` | ✅ |

---

## 2. Drift Pattern 总结

观察 Tasks #122 → #128 7 轮 housekeeping 显示 self-perpetuating drift pattern:

| Task | 修的 drift | 创造的 drift |
|------|----------|----------|
| #122 | TASKS_INDEX 漏 #113-#121 | (无 — refresh 当时同步) |
| #123 | REPRODUCE.md script 7 处不存在 | (无 — 文档对齐) |
| #124 | README/TASKS_INDEX 漏 #115-#123 | (无 — v2 refresh 同步) |
| #125 | CI workflow 4 separate → 1 dispatcher | (无 — dispatcher pattern 化) |
| #126 | TASKS_INDEX/CHANGELOG 漏 #124-#125 | 漏 #126 自己 |
| #127 | TASKS_INDEX/CHANGELOG 漏 #126 | 漏 #127 自己 |
| #128 | TASKS_INDEX/CHANGELOG 漏 #127 + §16 cleanup | (此 task 自己需后续清理) |

**根因**: TASKS_INDEX/CHANGELOG 是手维护文档, 闭环时不强制同步. 每个 task 闭环时 TASKS_INDEX 漏 1 个 entry → 下一个 task 必须做 "drift v(N+1)" 才能补齐.

**决策**: 暂停 housekeeping cycle (marginal value < 0).

---

## 3. 推荐 Auto-Gen 解决方案 (R11.3)

### scripts/task128_sync_task_docs.py spec (future task #129+)

```python
#!/usr/bin/env python3
"""Auto-generate TASKS_INDEX.md from descriptions/ + verdicts/."""
# 1. 扫 descriptions/task<N>_*.md + verdicts/task<N>_*.md (sorted by N)
# 2. 生成 TASKS_INDEX.md Coverage matrix + Recent closed tasks 段
# 3. diff vs 当前 TASKS_INDEX.md → 写新文件
# 4. exit 0 iff file changed (or --force 强制写)
```

Usage:
```bash
python3 scripts/task128_sync_task_docs.py        # 自动 diff + 写
python3 scripts/task128_sync_task_docs.py --check # exit 0 iff 同步
```

集成到 dispatcher (7th audit, future task #130+):
```python
("task129", "python3 scripts/task128_sync_task_docs.py --check"),
```

CI workflow: 不需要改 (dispatcher pattern 已支持).

**Task #128 = 推荐 spec**, 不实际实现 (over-engineering for current state).

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| §16 R8 cleanup 强度 | ✅ 强清 (删 R8 描述行 + 移活跃行) | ❌ 保留 (R8 明文禁止) |
| auto-gen TASKS_INDEX 是否实现 | ❌ 只 spec 不实现 | ✅ 实现 (但 over-engineering) |
| housekeeping cycle 是否继续 | ❌ 暂停 (drift 自我强化) | ✅ 继续 (边际价值 < 0) |
| §16 header 文本 | ✅ 改 Task #128 closure (新状态) | ❌ 保留 (过时) |

---

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| §16 表格 R8 cleanup | `grep -cE "R8 强制清理: Task #12[67]" loop.md` | ✅ 0 |
| §16 表格主体空白 | `grep -cE "^\| \(空" loop.md` | ✅ 1 |
| §16 header 含新 closure 描述 | `grep "housekeeping cycle closure" loop.md` | ✅ |
| dispatcher 6/6 PASS | `python3 scripts/all_audits.py` | ✅ 6/6 |

---

## 6. 关联

- 前置: Task #127 (6th audit + drift v3)
- 后置: 暂停 housekeeping cycle. Future task #129+ 推荐先实现 `scripts/task128_sync_task_docs.py` (auto-gen) 跳出 drift 循环

---

result: Task #128 — housekeeping cycle closure 闭环. §16 R8 强制清理 (删 #126/#127 活跃行 + 删 R8 描述行, 表格空白) + header 更新. drift pattern 总结 (self-perpetuating: 每个 task 创建 docs drift → 需 v2/v3 fix → 又创建新 drift). 推荐 auto-gen spec (scripts/task128_sync_task_docs.py, future task 实现). housekeeping cycle 暂停. dispatcher 6/6 PASS 不破坏.
