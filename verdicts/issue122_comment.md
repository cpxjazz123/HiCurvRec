## Issue #122 R18+R20 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 1 (= Stage 1 per-codeword gate + 训练): ❌ FAIL (5-step audit grad signal = 0)

**[5-step functional audit]**:
- gate_perturbed: ❌ False (loss 不变 — 只扰动 K0 alpha, 不影响其他 K 的 assignment)
- d_mix_changed_L0: ✅ True (K0 扰动后 d_mix 改变 — 验证 alpha → d_mix 因果)
- permuted_alpha: ❌ False (permute 不改 assignment if perm 是 convex 重组)
- **grad_finite_nz: ❌ False — grad_max_per_layer = [0.0, 0.0, 0.0]**
- roundtrip_ok: ✅ True
- entropy_per_layer: [1.099, 1.099, 1.099] (init 0 → uniform)

**失败原因 (核心发现)**:
- 5-step audit 中 grad_finite_nz 失败, grad 通过 gate_logits 实际为 **0.0**
- **架构根本问题**: HRQ-VAE forward 用 `argmin(d_mix)` + STE detach
- gate_logits 只通过 d_mix 进入 argmin, **没有进入 loss 路径**
- 跟 task414 (Issue #121) 同模式失败 (argmin + STE 切断 gradient)

**[后续 30 epoch 训练]**:
- 5-step audit FAIL 后, per spec "FAIL, 不训练", 直接 exit

**关键产物**:
- 实施脚本: `scripts/task415_issue122_codeword_conditional_gate.py` (~290 lines)
- per-codeword alpha_l,k = softmax(g_l,k), 3 components
- d_mix(x, c_k) = Σ_j alpha_l,k,j · d_l,j(x, c_k)
- 5-step audit (DIRECT perturbation 验证机制)

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL (5-step audit grad signal = 0)

### 跨方向联立 (R18 v2 4 维度)
- #115 + #116 + #118 + #119 + #121 + #122 全部 NO-GO 收口
- 共同根因 = **任何参数 (κ / gate) 通过 d_hyp/d_mix → argmin → STE detach → 没有 gradient 流入 loss**
- 需要 soft assignment (weighted sum by softmax(-d_mix/temp)) 才能让 gradient 流通

### 关键产物
- commit hash: 4399c5c
- push: origin/main
- verdict: verdicts/task415_issue122_gate1_fail_v2.md
- 整体决策: ❌ NO-GO 收口 (grad=0 架构问题)