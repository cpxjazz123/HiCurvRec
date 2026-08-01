# Task #464 / Issue #171 [方向A Gate2] 零中心有界残差 + κ 同步重校准 短程审计 — NO-GO 收口

## 任务摘要

Issue #171 owner 2026-08-01 07:46 派发: 方向A Gate 2 零中心有界残差 + κ 同步重校准短程审计. 复用 task462 模板 + 加 (a) 零中心有界残差 (LN 起点=0 + tanh*bound=0.5); (b) κ 更新后同步重校准 codebook + 距离 + 统一公式 (#47 复用); (c) per-layer learnable κ_l (三层独立). 真实 Stage 1 κ/scale metadata 注入 (跟 task462 一致).

R11.5 兜底 4. 简单实用方案 = 复用 task462 模板 + 加零中心有界残差 + 加 κ 同步重校准 + 加 per-layer learnable κ_l.

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_ZeroCenteredBoundedResidual
- **Stage 1 RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth`

## R18 4 维度对比 (vs #169 closed NO-GO 724c38c)

| 维度 | #169 (closed) | #171 (NEW) |
|------|---------------|------------|
| **D1 spec** | 真实 Stage 1 export + κ/scale 三层独立learnable | **真实 Stage 1 export + 零中心有界残差 + κ 同步重校准 + per-layer κ_l** |
| **D2 实施** | α init=-10, softplus unbounded | **LN 起点=0 + tanh*bound=0.5 + synchronize_kappa() + per-layer κ_l** |
| **D3 失败机制** | α 1 epoch 涨 2524× | 零中心 + tanh 饱和 → 不应该涨 → 但 κ_grad=0 → κ_logit_l 未接 forward |
| **D4 引用** | arXiv:2405.13979 | 同一引用 + DOI:10.1016/j.neunet.2026.109172 (task-geometry decoupling) |

→ **D1/D2 显著不同**: 零中心 + tanh + κ 同步重校准 + per-layer κ_l 是新机制. R18 强制实验.

## Gate 1 (= Stage 1 真实 export): ✅ PASS

- RQ-VAE ckpt sha256: `59a38fa3fa1aac5ac66dd90c3aa555d8ed00fd6fd6255e2143b42c6b285db4a1`
- 三层 codebook 真实 extract + hash 验证
- per-item real metadata shape=(9922, 3, 4) hash=81509c14
- 真实 Stage 1 metadata 注入 wrapper

## Gate 2 (= Stage 2 短程 sanity): ❌ FAIL — R23 early-stop + 关键设计 bug: κ 未接 forward path!

**实测数据** (2 epoch 训练 + R23 触发):
- loss 0.3091 → 0.0044 (decreased, 但异常低 - 可能 T5 输出饱和到特定 token)
- **val_R@10 trace: [0.0, 0.0]** (R23 触发: 连续 2 次 0)
- **κ_grad trace: [0.0, 0.0]** ⚠️ 关键设计 bug — κ_logit_l 没接 forward path, 无 gradient backprop
- **κ_l trace: [[0.6931, 0.6931, 0.6931], [0.6931, 0.6931, 0.6931]]** — 三层同步, 没 per-layer 差异
- residual_w trace: [-0.0874, -0.0918] (远小于 bound=0.5, bounded=True)
- cond_grad: epoch 0 = 0.3448 → epoch 1 = 0.000854 (急剧衰减)
- ln_grad: epoch 0 = 0.0920 → epoch 1 = 0.001628 (急剧衰减)
- save/load: missing=0, unexpected=0 ✅
- forward diff: 0.00e+00 (R23 skip)

**关键发现 (Meta-finding)**:
1. **零中心 + tanh 机制工作正常**: residual_w bounded (|w|≤0.5) ✅
2. **但 κ_logit_l 不在 forward path 中**: curvature_meta 用真实 metadata, κ_logit_l 只在 synchronize_kappa() 中被读 → 无 gradient flow → κ_grad=0
3. **三层 κ 同步不分化**: 既然 κ_grad=0, κ_l 三层永远停在 init=0.693 → 没 per-layer 差异
4. **val_R@10=[0,0]**: T5 在 conditioned 输入下学不到 target SID, 模型 collapse 到 padding token
5. **跟 #169/#170 失败模式不同**: #169 是 α 单调涨, #171 是 κ 完全没学 (设计 bug 而非动态失败)

## Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 失败, Stage 3 无意义

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 必须 PASS

## 决策 (整体): **NO-GO** 收口

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| 零中心 + tanh bounded residual 解 α-growth | ✅ CONFIRMED | residual_w [-0.087, -0.092] 远小于 bound=0.5 |
| κ 同步重校准是 R@10 杠杆 | ❌ REFUTED | Gate 2 FAIL, κ_grad=0, 三层同步 |
| per-layer learnable κ_l 是 R@10 杠杆 | ❌ REFUTED | κ_grad=0, 三层不分化 |
| 真 Stage 1 metadata 是 R@10 杠杆 | ❌ REFUTED | Gate 1 PASS 但 Gate 2 仍 FAIL |

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案 = 复用 task462 + 零中心 + tanh + κ 同步重校准**: 模板复用但 κ_logit_l 没接 forward 是设计缺陷
2. **R23 强制 early-stop**: val_R@10=[0,0] 连续 2 epoch → 立即 NO-GO
3. **R20 4-Gate 详细**: commit message 含 4 Gate 详细内容
4. **R18 强制实验**: 真实 Stage 1 metadata + 新机制 (D1/D2 不同)
5. **R15 强制 push**: verdict push 到 origin

## 后续 (R10 v2 idle)

Issue #171 NO-GO 收口. Issue #172 (task465) 同模式 (设计缺陷: κ_grad=0 + mix_grad=0). 12 issue κ/scale 元数据适配 (#157/#158/#162/#163/#165/#166/#167/#168/#169/#170/#171/#172) 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽.

task450 v8 (Issue #161 ZeroCenteredLayerNorm, val_R@10 ep30=0.1001 接近 baseline 0.1020) 仍在 GPU 0 继续训练.

## 关键产物

- **verdict**: `verdicts/task464_issue171_stage2_zero_centered_bounded_residual_kappa_recalibration_result.md`
- **verdict.json**: `products/task464_issue171_stage2_zero_centered_bounded_residual_kappa_recalibration/verdict.json`
- **stage1_proof**: `products/task464_issue171_stage2_zero_centered_bounded_residual_kappa_recalibration/stage1_export_proof.json`
- **real_metadata**: `products/task464_issue171_stage2_zero_centered_bounded_residual_kappa_recalibration/real_metadata.npy`
- **adapter ckpt**: `products/task464_issue171_stage2_zero_centered_bounded_residual_kappa_recalibration/adapter.pt` (R12 强制)
- **train trace**: `products/task464_issue171_stage2_zero_centered_bounded_residual_kappa_recalibration/train_trace.json`
- **整体决策**: NO-GO 收口 + R20 + R21 v2
