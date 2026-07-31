# Task #389 / Issue #96 [方向C Gate2] SID/metadata 多样性修复

**日期**: 2026-07-31
**触发**: Issue #96 [方向C Gate2] 先修复 SID/metadata 多样性再进 Stage 3 — R16 + R22 强制立即开工
**前置**: Issue #93 Gate 3 FAIL (commit bd9698a, shuffle_diff=0, 新根因: #87 SID collapse); Issue #43 HypPreEncoder 机制 PASS (task334)
**任务**: 1. R18 4 维度对比 vs Issue #87 #93; 2. FreeCurvHRQVAEWithHypPre Stage 1 训练 + Stage 2 Sinkhorn 推断 + 4-digit SID + metadata 重建; 3. 验证 SID unique ≥ 9500/9922 + metadata 方差非零; 4. Gate 2 PASS
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #96 vs Issue #87 #93)

| 维度 | Issue #87 (SID metadata FAIL) | Issue #93 (metadata warm-up FAIL) | Issue #96 (SID/metadata 多样性修复) |
|------|------|------|------|
| **D1 spec** | Gate 2 SID metadata 对齐 | Gate 3 metadata warm-up | Gate 2 先修 SID/metadata 多样性再进 Stage 3 |
| **D2 实施** | 复用 #86 ckpt + Sinkhorn | zero-init + warm-up + 500 steps | FreeCurvHRQVAEWithHypPre Stage 1 训练 + Stage 2 Sinkhorn |
| **D3 失败机制** | SID unique=256 坍缩 | metadata uniform → shuffle_diff=0 | HypPreEncoder + κ-Stereographic 修复 codebook collapse → SID 多样性 → metadata 多样性 |
| **D4 引用文献** | arXiv:2309.04082 | arXiv:2309.04082 | arXiv:2309.04082 (Curve Your Attention) |

→ 跟 #87 #93 完全不同方向, 必须做架构层实证.

## 2. Issue #96 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **Stage 1 HypPreEncoder** | FreeCurvHRQVAEWithHypPre 50 epoch 训练 | Issue #96 spec Gate2 |
| **Stage 2 Sinkhorn + dedup** | 4-digit SID 推断, 每层 use_sk=True | Issue #96 spec Gate2 |
| **metadata 重建** | per-layer kappa_l, scale_l, confidence, mask | Issue #96 spec Gate2 |
| **报告** | SID unique, collision rate, layer util, metadata per-field variance, top-k margin, κ/scale/conf 分布 | Issue #96 spec Gate2 |
| **PASS** | SID unique ≥ 9500/9922; collision ≤ 0.20; L0/L1/L2 util ≥ 90%; metadata per-field variance 非零 | Issue #96 spec Gate2 PASS |
| **FAIL** | SID unique 仍接近 #87 的 256, 或任一层 util < 90%, 或 metadata 方差接近 0 | Issue #96 spec Gate2 FAIL |

## 3. Issue #96 Gate 决策

| R17 Gate | 决策 | 状态 |
|------|------|------|
| Gate 1 (= Stage 1) | ✅ 复用 #86 PASS | 沿用 |
| **Gate 2 (= Stage 2 SID/metadata 多样性)** | ⏳ 进行中 | HypPreEncoder + Stage 2 Sinkhorn |
| Gate 3 (= Stage 3 T5 受控消融) | ⏸ STOP per spec | Gate 2 PASS 后 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Gate 1+2+3 PASS 后 |

## 4. 实施策略

- 复用 task387 同样的 FreeCurvHRQVAEWithHypPre wrapper
- Stage 1 训练 50 epoch (跟 task387 同结构)
- Stage 2: use_sk=True 跑 Sinkhorn, 收集 3-digit indices + 1 dedup column = 4-digit SID
- metadata 重建: kappa_l=layer.kappa_max*tanh(theta_m); scale_l=embeddings.weight.norm.mean; confidence=1/(1+margin); mask=1
- GPU 2 立即启动 (R7 + R19)

---

result: Issue #96 [方向C Gate2 SID/metadata 多样性修复] R22 + R19 立即开工 (GPU 2). 3 步: 1) FreeCurvHRQVAEWithHypPre Stage 1; 2) Stage 2 Sinkhorn + 4-digit SID + metadata 重建; 3) Gate 2 验证 + commit + push + comment(含 hash) → close. ⏳ 进行中.