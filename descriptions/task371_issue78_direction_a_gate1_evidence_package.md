# Task #371 / Issue #78 [方向A Gate1] κ-freeze warmup 证据包

**日期**: 2026-07-31
**触发**: Issue #78 [方向A Gate1] κ-freeze warmup 证据包 — R16 强制 GitHub OPEN 处理
**前置**: Issue #75 闭环 (task368 R18 实证 6/10, 实际 Stage 1 2 epoch 训练 + 全字段诊断)
**任务**: R18 4 维度 vs #75 对比 + 完整证据包 (config diff + optimizer 参数组 + freeze/unfreeze 切换点 + 日志字段清单 + 1-3 epoch 短跑 + 全字段)
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #78 vs Issue #75)

| 维度 | Issue #75 (Gate1 κ-freeze warmup 最小证据闭环) | Issue #78 (Gate1 κ-freeze warmup 证据包) | 是否一致 |
|------|------|------|------|
| **D1 spec 摘录** | 1-3 epoch 短跑 + 报告 warmup/unfreeze 全字段 | 证据包 = 配置 diff + optimizer 参数组 + freeze/unfreeze 切换点 + 日志字段清单 + verdict/commit 链接 + 短跑 | ❌ 不同 (Issue #78 要求完整证据集合) |
| **D2 实施核心** | LockableVQ θ_freeze + 两段 optimizer + 同步 scale + 实际训练循环 + 诊断 | 同 + 完整 config diff 输出 + 日志字段结构化 + 证据包 manifest | ❌ 不同 |
| **D3 Gate 1 失败机制** | 1-3 epoch 短跑, 验证机制真实运行 | 同 + 证据包要可审计可复现 | ❌ 不同 |
| **D4 引用文献** | arXiv:2405.13979v4 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). Issue #78 要求"证据包" = 完整证据集合 (配置 + 切换 + 日志 + 短跑 + 链接)**.

## 2. Issue #78 证据包要求

| 证据类型 | 内容 | 来源 |
|------|------|------|
| **配置 diff** | num_emb_list=[64,128,256], e_dim=32, M=3, kappa_max=2.0, layers=[512,256,128], loss_type='poincare', beta=0.25, sk_eps=[0.0,0.0,0.0] | FreeCurvHRQVAE __init__ |
| **optimizer 参数组** | warmup: encoder/decoder/codebook (lr=1e-3), theta_m lr=0; unfreeze: 全参数 lr=1e-3 | scripts/task371 Issue #78 实施 |
| **freeze/unfreeze 切换点** | epoch 0→1: theta_m.requires_grad False→True + codebook norm 同步归一化到 0.85 | freeze_theta_m / unfreeze_theta_m 函数 |
| **日志字段清单** | warmup/unfreeze 两段: κ_m, codebook_norm, distance_range, assignment_entropy, utilization, collision_rate, grad_theta_max | diagnose_layer 函数 |
| **verdict 路径** | verdicts/task371_issue78_evidence_package_v2.md | 本次 |
| **commit 链接** | 即将 push 的 commit hash | 即将 |

## 3. Issue #78 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 RQ-VAE/HRQVAE)** | ⏳ 进行中 | 证据包实施 + 1-3 epoch 短跑 + 全字段 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP (Gate 1 待 PASS) | per spec |
| Gate 3 (= Stage 3 T5-mini) | ⏸ STOP | per spec |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP | per spec |

---

result: Issue #78 [方向A Gate1 κ-freeze warmup 证据包] R18 3/4 维度 vs #75 不一致必须做实验. 要求完整证据集合 (config + 切换 + 日志 + 短跑 + 链接). ⏳ 进行中.