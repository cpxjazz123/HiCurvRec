# Task #261 — loop.md §16 状态同步 + 本轮 #257-#260 归档 (R8 强制清理)

## 背景

§16 表格内容跟仓库实际状态脱节 (2026-07-29 当前):
- **Task #243 epoch=200/400 训练**: §16 还写 "🏃 running, GPU 1 (util 96%) / GPU 2 (util 95%)", 但实际 PID 已不存在, GPU 0%, R12 ckpt 已存 (`HG_Rec_best.pth` 22MB at `products/task243/t5mini_epoch{200,400}/Instruments/Jul-29-2026_02-44-41/`), Stage 4 eval 未跑
- **本轮新增 Task #257-#260**: Issue #16 Gate 0/1 + Issue #10 Gate 0 + Sinkhorn 扫描 都没在 §16 表里登记

按 R8 强制清理规则, §16 表格主体应保持简洁 (当前真正在跑的任务或等用户决策的任务), 已闭环任务归档到 §17.

## 任务范围

零 GPU, 只改 loop.md §16 表格:
1. 移除 Task #243 "🏃 running" 行, 改 "☠️ dead (无 PID, GPU 0%)" + R12 ckpt 已存说明
2. 加本轮 Task #257-#260 到 "已闭环" 表 (#16 CLOSED, #10 Sinkhorn 扫描 FAIL)
3. 顶部状态段 (03:13 → 当前时间)
4. 列出"等用户决策"两个独立项 (Issue #10 方向 A1/A2/B/C + Task #243 Stage 4 eval 是否跑)

## 关键决策点 (R11.3)

- **Task #243 dead 但 R12 ckpt 已存**: 不删 products/ 目录 (按 R12 规则保留 R12 ckpt), 但 §16 不再 "running"
- **Task #243 Stage 4 eval**: 放 "等用户决策" 段, 让用户决定是否跑 (~2 min per ckpt eval)
- **Issue #10 方向选择**: 放 "等用户决策" 段, 提供新选项 A1 (β+Sinkhorn 联合扫描) / A2 (H1 不成立, 直接关) / B (等价 A2) / C (用户新方向)
- **scripts/task256 Arm B max_iters=20 launcher**: 在 §16 标注 "不建议启动" (Sinkhorn 20 在 vanilla 上等价 5/10/30)

## 物理产物

```
loop.md (仅 §16 表格修改, §1-§15 不动)
```

无 scripts/, 无新数据, 无新 verdict (Task #261 是 housekeeping, 不是研究).

## 未来 (等用户决策)

- 用户对 Issue #10 方向拍板 → 启动对应 launcher / 关闭 issue
- 用户对 Task #243 Stage 4 eval 拍板 → 跑 2 个 ckpt 的 Stage 4 (~2 min each, 跟 task253 stage4_eval.py 同模板)
- 下次 cron tick 按新 §16 继续推进

result: Task #261 — loop.md §16 表格状态同步完成. Task #243 running 行已改为 dead (R12 ckpt 已存, Stage 4 eval 等用户决策), 本轮 #257-#260 已加到 "已闭环" 表, Issue #10 方向 + Task #243 eval 列入 "等用户决策" 段. R8 强制清理规则执行完毕.