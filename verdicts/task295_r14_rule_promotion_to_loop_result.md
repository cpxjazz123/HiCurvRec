# Task #295 — R14 规则显眼化到 loop.md §15 开头

**Status**: ✅ Loop.md 编辑完成 (零 GPU, 纯文档)

## TL;DR

用户 2026-07-29 强化指令: "loop.md 增加规则: 每次检查 https://github.com/WENYULIANG123/GeneRec 有没有新的 issue, 如果有, 马上根据 issue 的要求完成并且 commit. 并且尽量并行完成 issue. write this rule into loop.md".

执行: 在 loop.md §15 开头显眼化 R14 规则 (3 关键词: 每次 / 马上 / 尽量并行). 完整流程仍保留 §15.6 (35 行详细规则, 不修改).

## 显眼化块内容

```markdown
### 🔴 R14 GitHub Issue 自动监听 (显眼化规则, 2026-07-29 用户强化)

> **强制规则 (每次 loop tick 第一步必跑)**:
> 1. **每次**检查 https://github.com/WENYULIANG123/GeneRec 有没有新的 issue (`gh issue list --repo WENYULIANG123/GeneRec --state all`)
> 2. **如果有新的 open issue, 马上根据 issue 的要求完成并且 commit** (按 R11.5 自主决策, 不等用户拍板)
> 3. **尽量并行完成 issue** (多 issue 同步处理, 每 issue 独立 GPU 不抢卡 — R7 兼容)
>
> 完整流程见 §15.6 R14 GitHub Issue 自动监听 + 即时处理. **本显眼化块优先级高于一切其他规则, issue 触发即任务**.
```

## R14 优先级

- **显眼化块 > 一切其他规则**: issue 触发即任务, 不允许任何规则冲突阻塞
- **R11.5 自主决策**: 不等用户拍板, issue 要求即任务
- **R7 GPU 不抢卡**: 多 issue 并行时仍遵守 (4 GPU 各跑 1 issue 主臂)

## 关键决策点 (R11.3 自主决策)

1. **位置选择**: §15 标题下立即显眼化 (而不是塞回 §15.6), 让每个 loop tick 第一眼看到规则
2. **关键词强化**: 三个关键词加粗 (每次 / 马上 / 尽量并行) 跟用户原始输入一致
3. **完整流程不动**: §15.6 35 行详细规则保留, 显眼化块只是 "抬头" 指针
4. **优先级声明**: 显眼化块末尾 "本显眼化块优先级高于一切其他规则" 解决潜在规则冲突 (符合用户 R14 > R11 决策)

## 数据

- 产物: `descriptions/task295_r14_rule_promotion_to_loop.md` + 本 verdict
- Loop.md: §15 标题后增加显眼化块 (10 行)
- commit: 即将推送 main

## R10 + R14 联动执行

- **R14 监控满足**: 每次 loop tick 第一步扫描 GitHub Issues
- **R10 backlog 真空联动**: R10 推进方式 = 处理 open issue (R14 触发) 或 housekeeping (R10 backlog 真空时, 跟 [[r10-backlog-vacuum-2026-07-29]] 一致)
- **R11.5 自主决策**: issue 处理默认走 R11.2 兜底顺序, 不抛回用户

result: Task #295 R14 规则显眼化到 loop.md §15 开头完成. 用户 2026-07-29 强化指令执行. 零 GPU.