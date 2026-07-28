# Task #234 — Issue #9 Hybrid Phase 1 Gate 0+1: per-layer assignment composition validation

## 来源
- GitHub Issue #9 (2026-07-29): [Hybrid Phase 1] Per-layer assignment: L0 = Per-Codeword κ c_k U(1,5) + L1/L2 = Gromov weight=0.5 (Task #231/#232 -> #233)
- Issue #9 §Hypothesis H1: hybrid 在串行 forward-pass 中三层是否仍 OPEN 60-90% (Phase 0 measured independently)
- Issue #9 §Hypothesis H2: Gromov 在 Stage 1 训练中会坍缩 (task221 证据), hybrid 是否继承坍缩

## 背景

Issue #6 (Task #231) + Issue #7 (Task #232) 单独测量的三层独立结果:

| Scheme | L0 | L1 | L2 | All-3 OPEN |
|--------|----|----|----|----|
| PC κ U(1,5) | 82.68% | 93.18% | 94.60% | no (L1/L2 near FAIL) |
| Gromov w=0.5 | 90.06% | 73.67% | 68.11% | **yes** |
| **proposed hybrid** | **82.68%** | **73.67%** | **68.11%** | yes (composition unverified) |

Issue #9 §Falsification: hybrid L0 perturbation ≈ +1.56pp (U(0.5,5)→U(1,5)) vs task220/222 endpoint R@10=0.0938, expect L0 ~20% utilization re-occur.

## 配置 (Issue #9 §Experimental design)

| Item | Value |
|------|-------|
| Gate 0 | per-layer assignment composition (CPU only, forward-pass) |
| Hybrid rule | L0 = PC κ c_k ~ U(1,5); L1/L2 = Gromov weight=0.5 |
| Phase 0 ckpt | HG-Rec baseline (#84) |
| Seeds | 3 (c_k_seed ∈ {42, 43, 44}) |
| L0 safe margin | agreement < 0.90 (PC κ 82.68% clears, Gromov 90.06% on boundary) |

## 决策阈值 (Issue #9 §Gates)

| Gate | 指标 | 阈值 | Pass 含义 | Fail 含义 |
|------|------|------|-----------|-----------|
| Gate 0 | composed 3-layer forward-pass | 三层 60-90% OPEN, 与独立测量 ±3pp | H1 成立 (hybrid 不破坏 OPEN) | H1 false, "首个三层全 OPEN" 不成立 |
| Gate 1 | Stage 1 collision + L0 util | collision ≤ 0.3706 + L0 util ≥ 90% | 训练中不坍缩 | 继承 Gromov 坍缩, 方向关闭 |
| Gate 2 | Stage 2 unique SID | > 6245 (62.94%) | codebook 健康 | codebook 退化 |
| Gate 3 | Stage 4 R@10 | ≥ 0.1058 (vanilla + Sinkhorn) | 值得采纳 | NO-GO (跟 task225 持平) |

## 步骤 (Issue #9 §Procedure)

1. ✅ 创建 task234 scripts (扩展 task231/232 Phase 0 脚本接受 per-layer assignment)
2. 跑 Gate 0 (3 seeds, composed vs per-layer-independent consistency)
3. Gate 0 pass → Stage 1 40 epoch 训练 (R12 + per-5-epoch logging)
4. Gate 1 评估 vs task222 best row
5. Gate 1 pass → Stage 2 SID + Stage 3+4 R@10
6. 写 verdict with collision/utilization reported per epoch alongside every recall number

## 产物

- products/task234/hybrid_gate0_3seeds.json (Gate 0 composed 数字)
- products/task234/phase1_arm_hybrid/ (Gate 1 ckpt if pass)
- verdicts/task234_hybrid_gate0_result.md
- verdicts/task234_hybrid_gate1_result.md (if Gate 0 pass)

## 依赖

- 复用 task231/232 helpers + task218/219 baseline ckpt
- 复用 task222 Stage 1 协议 (40 epoch early_stop)
- Issue #8 verdict 影响 Gate 3 pass bar (待 3-arm converged collision design)

## Status
Not yet run. Issue #9 Gate 0 (cheap, minutes CPU) + Gate 1 (~30s GPU) 应该立即跑. Gates 2-3 不调度直到 Gate 1 pass.

R11.3 自主决策:
- **架构 baseline** = Hybrid L0=PC κ U(1,5) + L1/L2=Gromov w=0.5 (Issue #9 提议, 不是 R11.3 备选)
- **备选 A**: 仅 L0=PC κ + L1/L2=Euclidean baseline (跟 HG-Rec #84 baseline 一样的 fallback)
- **备选 B**: 三层都用 Gromov w=0.5 (纯 Gromov)
