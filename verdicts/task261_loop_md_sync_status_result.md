# Task #261 — loop.md §16 状态同步 + 本轮 #257-#260 归档 + R8 清理

## 关键变更

| §16 内容 | 旧 | 新 |
|---|---|---|
| 顶部状态段 | 2026-07-29 03:13 AEST (Task #243 running GPU 1/2) | 2026-07-29 当前 (Issue #10 仍 OPEN + Sinkhorn FAIL, Issue #16 CLOSED, 4 GPU 全空闲) |
| Task #243 epoch=200 行 | 🏃 running, GPU 1 util 96%, epoch 34/200 | ☠️ dead (无 PID, GPU 0%), R12 ckpt 已存, Stage 4 eval 未跑 |
| Task #243 epoch=400 行 | 🏃 running, GPU 2 util 95%, epoch 34/400 | ☠️ dead (同上, 路径 epoch=400/) |
| #10 行 | Task #236/237/245 collision 口径 + 3-arm | Task #236/237/245/259/260 collision 口径 + 3-arm + Sinkhorn 扫描 (Gate 1 Sinkhorn FAIL) |
| 新增 #13 行 | (缺) | Task #248/249/253/254 Möbius 残差 4-gate NO-GO (证据已退回 task225) |
| 新增 #16 行 | (缺) | Task #257/258 #13 Gate 2 越闸 audit ✅ CLOSED |
| 等用户决策 | 1 段 (Task #238 Issue #10 redesign 4-arm) | 2 段: (a) Issue #10 方向 A1/A2/B/C (修订自原 Task #238 4-arm), (b) Task #243 Stage 4 eval |

## 关键决策点 (R11.3)

- **不删 Task #243 products/**: R12 ckpt 保留 (按 R12 强制规则), 但 §16 不再写 "running"
- **Issue #10 方向重新设计**: Sinkhorn 旋钮无效 (#260 evidence) → 修订原 4-arm Task #238 提议为 A1 (β+Sinkhorn 联合) / A2 (H1 不成立, 直接关) / B / C 四选项
- **Task #243 Stage 4 eval 列入用户决策**: R12 ckpt 已存, eval 跑一下只需 ~2 min × 2, 但需要用户决策是否值得跑 (Task #243 原始动机是 task233 budget-fix control arm, 跟 Issue #10 推进正交)
- **scripts/task256 Arm B max_iters=20 launcher 标记 "不建议启动"**: #260 Sinkhorn 扫描证明 Sinkhorn 20 在 vanilla 上等价于 Sinkhorn 5/10/30, 跑 Arm B 不会改变 issue 结论

## 物理产物

```
descriptions/task261_loop_md_sync_status.md
verdicts/task261_loop_md_sync_status_result.md  (本文件)
loop.md  (仅 §16 表格修改)
```

## 后续

- 用户对 Issue #10 方向拍板 (A1/A2/B/C)
- 用户对 Task #243 Stage 4 eval 拍板 (跑/不跑)
- 下次 cron tick 按新 §16 状态继续推进

result: Task #261 — loop.md §16 状态同步完成, R8 强制清理规则执行完毕. Task #243 改 dead + R12 ckpt 已存标注, 本轮 #257-#260 已加到 "已闭环" 表, 顶部状态段 + "等用户决策" 段更新.