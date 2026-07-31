# Task #368 / Issue #75 [方向A Gate1] κ-freeze warmup 最小证据闭环

**日期**: 2026-07-31
**触发**: Issue #75 [方向A Gate1] κ-freeze warmup 最小证据闭环 — R16 强制 GitHub OPEN 处理
**前置**: Issue #72 闭环完成 (task365 旧 verdict, R18 实证, 但仅 precheck + placeholder launcher)
**任务**: R18 4 维度 vs #72 对比 + 实际 Stage 1 1-3 epoch 训练 + 报告 warmup/unfreeze 两段 κ/scale/codebook norm/distance/assignment entropy/utilization/collision_rate/梯度范数
**结果**: ⏳ 进行中 (task381 跟踪)

---

## 1. R18 4 维度对比 (Issue #75 vs Issue #72)

| 维度 | Issue #72 (Gate1 κ-freeze warmup 预检) | Issue #75 (Gate1 最小证据闭环) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | precheck 5/5 PASS + R19 启动 placeholder | 实际 Stage 1 1-3 epoch + 报告 warmup/unfreeze 全字段 | ❌ 不同 (Issue #75 要求实际 GPU 训练 + 完整字段) |
| **D2 实施核心** | placeholder launcher + precheck 静态审计 | LockableVQ θ_freeze + 两段 optimizer + 同步 scale + Stage 1 RQ-VAE 实际训练循环 + 输出全字段 | ❌ 不同 |
| **D3 Gate 1 失败机制** | precheck 仅静态审计, 无真实训练 | 假设 warmup 阶段 freeze θ 后 unfreeze 同步 scale 可解 collapse | ❌ 不同 |
| **D4 引用文献** | arXiv:2405.13979v4 | arXiv:2405.13979v4 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #75 要求实际 GPU 训练 (vs #72 仅 placeholder)**.

## 2. Issue #75 关键字段 (per spec)

- 必须报告 warmup/unfreeze 两段的: κ, scale, codebook norm, distance range, assignment entropy, utilization, collision_rate, 梯度范数
- PASS 标准: L0/L1/L2 utilization ≥90%, collision_rate ≤0.20, 无 NaN/Inf, theta/scale 梯度有限非零
- 预检 PASS 需提交: 配置 diff + optimizer 参数组 + freeze/unfreeze 切换点 + 日志字段清单 + verdict/commit 链接
- Gate2/3/4 禁止执行, 直到 Gate1 有外部可核验 PASS 证据

## 3. Issue #75 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏳ **进行中** | 1-3 epoch 实证 + 全字段报告 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Gate 1 待 PASS) | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

---

result: Issue #75 [方向A Gate1 κ-freeze warmup 最小证据闭环] R18 3/4 维度 vs #72 不一致必须做实验. 要求实际 Stage 1 1-3 epoch 训练 + 全字段报告. ⏳ 进行中 (task381 跟踪).