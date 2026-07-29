# Task #295 — R14 规则显眼化到 loop.md §15 开头 (2026-07-29 用户强化)

## 背景

用户 2026-07-29 强化指令: "loop.md 增加规则: 每次检查 https://github.com/WENYULIANG123/GeneRec 有没有新的 issue, 如果有, 马上根据 issue 的要求完成并且 commit. 并且尽量并行完成 issue. write this rule into loop.md"

R14 已存在 §15.6 (35 行详细规则), 但用户要求"显眼化", 提升到 §15 开头, 强调:
1. **每次** loop tick 第一步必跑 (不是"建议", 是"强制")
2. **马上**根据 issue 要求完成并且 commit (不等用户拍板)
3. **尽量并行**完成 issue (多 issue 同步处理, GPU 不抢卡)

## 实施

1. 在 §15 标题后立即增加显眼化块 `🔴 R14 GitHub Issue 自动监听 (显眼化规则, 2026-07-29 用户强化)`
2. 强调三个关键词: 每次 / 马上 / 尽量并行
3. 完整流程保留 §15.6 (不动)
4. 显眼化块优先级: "本显眼化块优先级高于一切其他规则, issue 触发即任务"

## 产物

- `descriptions/task295_r14_rule_promotion_to_loop.md` (本文件)
- `verdicts/task295_r14_rule_promotion_to_loop_result.md` (closeout)
- `loop.md` §15 开头显眼化块追加 (commit + push)
- `memory/r14-rule-prominent-loop.md` (R10 决策记录)

## 不消耗 GPU

零 GPU, 纯 loop.md 编辑 + commit. 预计 5-10 分钟完成.

## R14 详细规则仍生效 (不修改 §15.6)

§15.6 仍包含 35 行详细规则 (实施细节 / 与现有规则的关系 / commit 粒度 / issue 处理流程 7 步等). 本任务只增加显眼化块, 不修改 §15.6 内容.

result: Task #295 R14 规则显眼化到 loop.md §15 开头. 用户 2026-07-29 强化指令执行. 零 GPU.