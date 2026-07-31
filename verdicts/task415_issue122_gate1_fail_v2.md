# Task #415 / Issue #122 [方向B Gate1] Gate 1 FAIL 收口 verdict

**日期**: 2026-08-01
**Issue**: #122 [方向B Gate1] — per-codeword alpha_l,k 替代全局 alpha, 防 L1/L2 坍缩放大
**任务**: CodewordGateHRQVAE K=[64,128,256] 30 epoch main (per-codeword gate)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 per-codeword gate + 训练): ❌ FAIL (5-step audit grad signal = 0)

**[5-step functional audit]**:
- gate_perturbed: ❌ False (loss 不变 — 只扰动 K0 alpha, 不影响其他 K 的 assignment)
- d_mix_changed_L0: ✅ True (K0 扰动后 d_mix 改变 — 验证 alpha → d_mix 因果)
- permuted_alpha: ❌ False (permute 不改 assignment if perm 是 convex 重组)
- grad_finite_nz: ❌ **False — grad_max_per_layer = [0.0, 0.0, 0.0]**
- roundtrip_ok: ✅ True
- entropy_per_layer: [1.099, 1.099, 1.099] (init 0 → uniform, 熵 = ln(3))

**失败原因 (核心发现)**:
- 5-step audit 中 grad_finite_nz 失败, grad 通过 gate_logits 实际为 **0.0**
- **架构根本问题**: 当前 HRQ-VAE forward 用 `argmin(d_mix)` + STE (`z_q_st = z_e + (z_q - z_e).detach()`)
- 梯度路径: Loss → decoder(z_q) → z_q = codebook[assign] → argmin (non-diff) → STE detach
- gate_logits 只通过 d_mix 进入 argmin, **没有进入 loss 路径**
- 因此 gate_logits 没有任何 gradient signal, 训练期 optimizer.step() 不会改变 alpha
- 跟 task414 同模式失败 (Issue #121 同样 grad=0)

**[后续 30 epoch 训练]**:
- 5-step audit FAIL 后, per spec "FAIL, 不训练", 直接 exit

**关键产物**:
- 实施脚本: `scripts/task415_issue122_codeword_conditional_gate.py` (~290 lines)
  - CodewordGateLayer: per-codeword alpha_l,k = softmax(g_l,k), 3 components
  - d_mix(x, c_k) = Σ_j alpha_l,k,j · d_l,j(x, c_k)
  - 5-step functional audit (DIRECT perturbation 验证机制)
  - 30 epoch main (未启动, 因 audit FAIL)
- 路径差异 vs #119: 实施路径**有差异** (per-codeword vs 全局 alpha), 但根因同源 (argmin + STE detach 让 gradient 不通过 alpha)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad signal = 0)

---

## 跨方向联立 (R18 实证 4 维度)

| 维度 | Issue #119 (task412, closed) | Issue #122 (本 task) |
|------|------------------------------|----------------------|
| **D1 spec 摘录** | d_mix = Σ_j α_lj d_lj (全局) | **d_mix = Σ_j α_l,k,j d_l,j (per-codeword)** ✅ |
| **D2 实施核心** | alpha 是 layer 标量 (per-layer) | **alpha 是 per-layer per-codeword (K 个)** ✅ |
| **D3 Gate 失败机制** | 30 epoch 短训 mode collapse | **grad through alpha = 0 (架构问题)** ✅ |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2307.04514 ✅ |

**R18 v2 强制结论**: Issue #122 路径**有差异** (per-codeword vs 全局). 实施后发现根因更深层: **HRQ-VAE 的 argmin + STE 设计让 gate_logits 没有 gradient 流入 loss 路径**. 即使 d_mix_changed_L0=True (alpha → d_mix 因果链成立), gate_logits 也不会因为 loss 优化而改变.

**联立 #115 + #116 + #118 + #119 + #121 + #122**: 6 个 issue 全部 NO-GO 收口, 共同根因 = **任何参数 (κ / gate) 通过 d_hyp/d_mix → argmin → STE detach → 没有 gradient 流入 loss**. 需要 soft assignment (weighted sum by softmax(-d_mix/temp)) 才能让 gradient 流通, 但这是架构变更, 超出本任务范围.

---

## R11.5 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| alpha_l,k 参数化 | alpha_l,k = softmax(g_l,k), g_l,k learnable | Issue spec §Gate1 1 强制 |
| alpha 维度 | (L, K, 3) — 每层每 codeword 3 个 component 权重 | Issue spec §Gate1 1 强制 |
| d_mix 公式 | d_mix(x, c_k) = Σ_j alpha_l,k,j · d_l,j(x, c_k) | Issue spec §Gate1 1 强制 |
| Audit 扰动方式 | DIRECT perturbation gate_logits[0] = [-3, 0, +5] | asymmetric 避免 softmax offset invariance |
| grad threshold | 1e-20 | grad_max=0.0 即使阈值=1e-20 仍 FAIL |
| 训练 epoch | 30 main | Issue spec "相同预算" |
| GPU 分配 | GPU 1 | R7 + R19 跨 issue 并行 |
| Conda env | `genrec_env` (R1 默认) | R1 v1 通用 GPU/CPU 计算任务 |
| Gate 1 决策 | ❌ FAIL (grad=0) | 机制验证诚实记录, 不允许 fudge |

---

## 整体决策

**❌ NO-GO 收口 (5-step audit FAIL: grad through gate_logits = 0)**

Issue #122 per-codeword alpha 设计数学正确 (d_mix 因果链成立), 但 HRQ-VAE 架构让 gate_logits 失去 gradient path. 任何 alpha 参数化变体 (per-layer / per-codeword / per-component) 都无法绕开 argmin + STE detach 的根本问题.

**Issue #122 closed NO-GO** (per R16 + R20 + R21).

---

## 后续 (per R22 + R19 + R16)

1. ✅ 写 Gate 1 FAIL verdict (本文件)
2. ⏳ commit + push (R15 + R21 v2)
3. ⏳ gh issue close #122 --reason completed (R16)
4. ⏳ gh issue comment #122 含 4 Gate 详细 + commit hash (R20 + R21)