# Task #387 / Issue #94 [方向A Gate1] HypPreEncoder + κ-Stereographic 距离公式准入验证

**日期**: 2026-07-31
**触发**: Issue #94 [方向A Gate1] HypPreEncoder κ-Stereographic 距离公式准入验证 — R16 + R22 强制立即开工
**前置**: Issue #91 Gate 1 FAIL (commit 0ef9eb9, step1 max_load=82.56%); Issue #43 HypPreEncoder 机制 PASS (commit 0cbe607, task334)
**任务**: 1. R18 4 维度对比 vs Issue #91; 2. FreeCurvHRQVAEWithHypPre 接入 HypPreEncoder (task334 wrapper) BEFORE encoder; 3. 50 epoch 训练验证 step1 不再 collapse; 4. Gate 1 PASS 验证
**结果**: ⏳ 进行中

---

## 1. R18 4 维度对比 (Issue #94 vs Issue #91)

| 维度 | Issue #91 (kmeans_init + β=0 修复 FAIL) | Issue #94 (HypPreEncoder + κ-Stereo 准入验证) | 一致? |
|------|------|------|------|
| **D1 spec** | kmeans_init=True + β=0 first epoch 修复 | HypPreEncoder + κ-Stereographic 距离公式准入 (架构层修复) | ❌ |
| **D2 实施** | FreeCurvHRQVAE + kmeans_init + β=0 | FreeCurvHRQVAEWithHypPre (复刻 task334 wrapper, base=FreeCurvHRQVAE) | ❌ |
| **D3 失败机制** | 假设 commitment loss 根因 → β=0 修复 | 假设输入端几何失真 → expmap0 把 Euclidean 输入映射到 Poincaré ball, 让 encoder 不"扁平化" hyperbolic 信号 | ❌ (新方向) |
| **D4 引用文献** | arXiv:2405.13979 | arXiv:2405.13979 (curvature-aware optimization) | ✅ |

→ R18 3/4 维度不一致, 必须做架构层实证 (R18 强制).

## 2. Issue #94 spec 要求

| 要求 | 内容 | 来源 |
|------|------|------|
| **HypPreEncoder 准入** | 输入 768d → expmap0 (c=0.74) → encoder | Issue #94 spec Gate1 |
| **κ-Stereographic 距离公式** | 复用 issue #47 统一公式 (per-component dist) | Issue #94 spec Gate1 |
| **三层 learnable θ/κ 路径** | L0 K64 / L1 K128 / L2 K256, 每层独立 θ_l, κ_l | Issue #94 spec Gate1 |
| **报告** | step1/step50/final L0/L1/L2 max_load, util, top-k margin, pairwise dist, codebook norm, θ/κ, NaN/Inf | Issue #94 spec Gate1 |
| **PASS** | step1 max_load < 50%; final L0/L1/L2 util ≥ 90%; SID collision ≤ 0.20 | Issue #94 spec Gate1 PASS |
| **FAIL** | 任一层 step1 collapse_50%, 或 final util < 90% | Issue #94 spec Gate1 FAIL |

## 3. Issue #94 Gate 决策 (R17/§19)

| R17 Gate | 决策 | 状态 |
|------|------|------|
| **Gate 1 (= Stage 1 HypPreEncoder 验证)** | ⏳ 进行中 | FreeCurvHRQVAEWithHypPre 50 epoch 训练 |
| Gate 2 (= Stage 2 Sinkhorn) | ⏸ STOP per spec | Gate 1 PASS 后 |
| Gate 3 (= Stage 3 T5) | ⏸ STOP per spec | Gate 1+2 PASS 后 |
| Gate 4 (= Stage 4 R@K eval) | ⏸ STOP per spec | Gate 1+2+3 PASS 后, R@10 > 0.1020 |

## 4. 实施策略 (R11.5 自主决策)

- 复刻 task334 HRQVAEWithHypPre wrapper, base 改为 FreeCurvHRQVAE
- HypPreEncoder: c=0.74 (Issue #43 Task #70 Ollivier mean), enabled=True
- 跟 task384 同样 kmeans_init=True + 50 epoch 训练 + per-step trace
- GPU 0 立即启动 (R7 + R19 激进, 4 卡空闲)

---

result: Issue #94 [方向A Gate1 HypPreEncoder + κ-Stereographic 准入验证] R22 + R19 立即开工 (GPU 0). 3 步: 1) FreeCurvHRQVAEWithHypPre; 2) 50 epoch 训练 + per-step trace; 3) Gate 1 验证 + commit + push + comment(含 hash) → close. ⏳ 进行中.