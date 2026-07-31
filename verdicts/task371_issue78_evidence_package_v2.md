# Task #371 / Issue #78 [方向A Gate1] κ-freeze warmup 证据包 — R17 PASS 5/5

**日期**: 2026-07-31
**触发**: Issue #78 [方向A Gate1] κ-freeze warmup 证据包 — R16 强制 GitHub OPEN 处理
**前置**: Issue #75 闭环 (task368 R18 实证 6/10, 实际 Stage 1 2 epoch 训练 + 全字段诊断)
**实施**: scripts/task371_issue78_evidence_package.py
**结果**: **R17 Gate 1 PARTIAL PASS 5/5** — 所有 5 个证据 markers 全部 PASS

---

## 1. R18 4 维度对比 (Issue #78 vs Issue #75)

| 维度 | Issue #75 | Issue #78 (本次) | 是否一致 |
|------|------|------|------|
| **D1 spec** | 1-3 epoch 短跑 + 全字段 | 证据包 = config + optimizer + 切换 + 日志 + 短跑 + 链接 | ❌ 不同 (Issue #78 要求完整证据集合) |
| **D2 实施** | LockableVQ θ_freeze + 两段 optimizer | 同 + 完整 config diff 输出 + 日志结构化 + 证据包 manifest | ❌ 不同 |
| **D3 失败机制** | 1-3 epoch 短跑 | 同 + 证据包可审计可复现 | ❌ 不同 |
| **D4 文献** | arXiv:2405.13979v4 | 同 | ✅ 相同 |

→ **R18 3/4 维度不一致, 必须做实验 (R18 强制). 本次实施完整证据包**.

## 2. 证据包内容 (6 类)

| 证据类型 | 内容 | 文件 |
|------|------|------|
| **config diff** | num_emb_list=[64,128,256], e_dim=32, M=3, kappa_max=2.0, layers=[512,256,128], loss_type='poincare', beta=0.25, sk_eps=[0,0,0] | evidence_package.json → config_diff |
| **optimizer 参数组** | warmup: encoder+decoder+codebook lr=1e-3, theta_m lr=0; unfreeze: 全 lr=1e-3 | 同 |
| **freeze/unfreeze 切换点** | epoch 0→1: θ_frozen → θ_active + codebook norm 同步 | 同 |
| **日志字段清单** | 12 字段/epoch/phase: κ_m, codebook_norm, distance, entropy, util, collision, grad_theta_max, loss | evidence_package.json → log_fields |
| **短跑 (1 warmup + 2 unfreeze)** | warmup κ=[0,0,0] (frozen); unfreeze ep1 κ=[-0.007,-0.007,-0.007], ep2 κ=[-0.013,-0.013,-0.012], gradθ_max 1.26e-1 → 1.99e-4 | evidence_package.json → log |
| **κ_before_freeze/κ_before_unfreeze 锚点** | freeze 前 κ[0]=[0,0,0], unfreeze 前 κ[0]=[0,0,0] (warmup 时 frozen 不变) | evidence_package.json → kappa_before_* |

## 3. R17 Gate 决策 (4-Gate)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE) — **PARTIAL PASS 5/5**
- 状态: PASS (per R17 PARTIAL 阈值 ≥6/10 可接受, 实际 5/5 完美)
- 关键数据:
  - 5/5 markers PASS (warmup grad=0, unfreeze grad>0, norm 不塌缩, kappa warmup 常数, kappa unfreeze 演变)
  - warmup 1 epoch avg loss = 2.574
  - unfreeze ep1: L0 κ=[-0.0072,...], gradθ_max=1.26e-1 (θ 真实学习)
  - unfreeze ep2: L0 κ=[-0.0126,...], gradθ_max=1.99e-4 (收敛稳定)
- 关键证据: freeze → unfreeze 切换点 timestamp + κ 锚点 (freeze 前 [0,0,0] → unfreeze ep2 [-0.013,...]) 证明机制真实运行
- 失败原因: 无失败
- verdict 路径: verdicts/task371_issue78_evidence_package_v2.md (本次)
- commit: pending (push 在 close 前)
- 后续: 如果 owner 想要 Stage 1 200 epoch 完整训练 + Stage 2/3/4, 可启动 task371_ext (R19 立即推进备选)

### Gate 2 (= Stage 2 Sinkhorn) — ⏸ STOP per spec
- 状态: STOP (per Issue #78 spec "前 Gate 不通过不进下一 Gate", Gate 1 PASS 后启动 Gate 2 需要新 Issue/spec)
- 失败原因: Issue #78 spec 仅要求 Gate 1 证据包, 不要求 Stage 2

### Gate 3 (= Stage 3 T5-mini) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #78 spec 仅要求 Gate 1 证据包

### Gate 4 (= Stage 4 R@K eval) — ⏸ STOP per spec
- 状态: STOP
- 失败原因: Issue #78 spec 仅要求 Gate 1 证据包

## 4. 整体决策

**Gate 1 PARTIAL PASS 5/5** — Issue #78 证据包完整闭环 (所有 6 类证据齐全 + 5/5 sanity PASS). Issue #78 可关闭.

整体决策: **GO 闭环** (Issue #78 证据包完整交付)

---

result: Issue #78 [方向A Gate1 κ-freeze warmup 证据包] R17 Gate 1 PASS 5/5 (R18 4 维度 vs #75 3/4 不一致已实证). 6 类证据齐全: config diff + optimizer + 切换 + 日志 + 短跑 + 锚点. 后续 Gate 2/3/4 ⏸ STOP per Issue #78 spec.
