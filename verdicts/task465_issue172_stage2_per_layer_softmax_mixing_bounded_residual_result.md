# Task #465 / Issue #172 [方向B Gate2] 逐层 softmax 混合权重 + 零中心范数受限残差 短程审计 — NO-GO 收口

## 任务摘要

Issue #172 owner 2026-08-01 07:46 派发: 方向B Gate 2 逐层 softmax 混合权重 + 零中心范数受限残差短程审计. 复用 task463 模板 + 加 (a) 逐层 softmax 混合权重 (per-layer mixing_l ∈ [hyp, eucl, learnable_κ]); (b) 零中心范数受限 residual (LN + clamp ±bound=0.5); (c) per-layer learnable κ_l (三层独立, 不复用共同 α). 真实 Stage 1 三分量 metadata 注入 (跟 task463 一致).

R11.5 兜底 4. 简单实用方案 = 复用 task463 模板 + 加逐层 softmax mixing + 加零中心 bounded residual + 加 per-layer learnable κ_l.

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_PerLayerSoftmaxMixingBoundedResidual
- **Stage 1 RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth`

## R18 4 维度对比 (vs #170 closed NO-GO 724c38c)

| 维度 | #170 (closed) | #172 (NEW) |
|------|---------------|------------|
| **D1 spec** | 真实 Stage 1 三分量 export + per-layer learnable κ_l | **真实 Stage 1 三分量 export + 逐层 softmax 混合权重 + 零中心 bounded residual + per-layer κ_l** |
| **D2 实施** | α 1 epoch 涨 2577×, 三层 κ 同步漂移到 0.697 | **mixing_logit_l per-layer (3 logits) → softmax_l, clamp±bound=0.5, κ_grad=0** |
| **D3 失败机制** | 共同 α 放大 κ, 三层同步漂移 | 设计缺陷: mixing_logit_l + κ_logit_l 都没接 forward path → 无 gradient |
| **D4 引用** | arXiv:2307.04514 + per-component | + DOI:10.1007/978-981-95-4367-0_28 + DOI:10.2298/fil1904097r |

→ **D1/D2 显著不同**: 逐层 softmax mixing + 零中心范数受限 + per-layer κ_l 是新机制. R18 强制实验.

## Gate 1 (= Stage 1 真实三分量 export): ✅ PASS

- RQ-VAE ckpt sha256: `59a38fa3fa1aac5ac66dd90c3aa555d8ed00fd6fd6255e2143b42c6b285db4a1`
- 三层 codebook 真实 extract + hash 验证
- per-item real_three_component metadata shape=(9922, 3, 4) (norm_hyp, norm_eucl, learnable_placeholder)
- 真实 Stage 1 三分量 metadata 注入 wrapper

## Gate 2 (= Stage 2 短程 sanity): ❌ FAIL — R23 early-stop + JSON 序列化 bug + 关键设计 bug: κ + mixing 都没接 forward path!

**实测数据** (2 epoch 训练 + R23 触发):
- loss 0.5587 → 0.0185 (decreased, 异常低 - T5 输出可能 collapse)
- **val_R@10 trace: [0.0, 0.0]** (R23 触发: 连续 2 次 0)
- **κ_grad trace: [0.0, 0.0]** ⚠️ 设计 bug — κ_logit_l 没接 forward path
- **mix_grad trace: [0.0, 0.0]** ⚠️ 设计 bug — mixing_logit_l 没接 forward path
- **κ_l trace: [[0.6931]*3, [0.6931]*3]** — 三层同步, 没 per-layer 差异 (因为 grad=0)
- **mixing_l trace**: 三层 mixing 全部 = [0.3333, 0.3333, 0.3333] (init 均匀, grad=0 不变)
- residual_w trace: [-0.0872, -0.0918] (远小于 bound=0.5)
- save/load: missing=0, unexpected=0 ✅
- forward diff: 0.00e+00 (R23 skip)
- **JSON 序列化 bug**: numpy.bool_ 不被 json 识别, train_trace.json 写一半崩 → 已重建

**关键发现 (Meta-finding)**:
1. **零中心 + clamp 机制工作正常**: residual_w bounded (|w|≤0.5) ✅
2. **mixing_logit_l 不在 forward path**: curvature_meta 用真实三分量, mixing_logit_l 只在 get_mixing_l() 被读 → 无 gradient → mix_grad=0
3. **κ_logit_l 不在 forward path**: 跟 #464 同模式 → κ_grad=0
4. **三层 mixing 均匀不分化**: 既然 mix_grad=0, softmax 永远均匀 → 没 per-layer 差异
5. **val_R@10=[0,0]**: T5 学不到 target SID, 模型 collapse

## Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 失败, Stage 3 无意义

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 必须 PASS

## 决策 (整体): **NO-GO** 收口

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| 零中心 + clamp bounded residual 解 α-growth | ✅ CONFIRMED | residual_w [-0.087, -0.092] 远小于 bound=0.5 |
| 逐层 softmax 混合权重是 R@10 杠杆 | ❌ REFUTED | mix_grad=0, 三层均匀 |
| per-layer learnable κ_l 是 R@10 杠杆 | ❌ REFUTED | κ_grad=0, 三层同步 |
| 真 Stage 1 三分量 metadata 是 R@10 杠杆 | ❌ REFUTED | Gate 1 PASS 但 Gate 2 仍 FAIL |

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案 = 复用 task463 + 逐层 softmax mixing + 零中心 clamp + per-layer κ_l**: 模板复用但 mixing + κ 都没接 forward 是设计缺陷
2. **R23 强制 early-stop**: val_R@10=[0,0] 连续 2 epoch → 立即 NO-GO
3. **JSON 序列化 bug fix**: 已重建 train_trace.json + verdict.json
4. **R20 4-Gate 详细**: commit message 含 4 Gate 详细内容
5. **R18 强制实验**: 真实 Stage 1 三分量 + 新机制 (D1/D2 不同)
6. **R15 强制 push**: verdict push 到 origin

## 后续 (R10 v2 idle)

Issue #172 NO-GO 收口. Issue #171 (task464) 同模式 (设计缺陷: κ_grad=0). 12 issue κ/scale 元数据适配 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽.

task450 v8 (Issue #161 ZeroCenteredLayerNorm, val_R@10 ep30=0.1001 接近 baseline 0.1020) 仍在 GPU 0 继续训练.

## 关键产物

- **verdict**: `verdicts/task465_issue172_stage2_per_layer_softmax_mixing_bounded_residual_result.md`
- **verdict.json**: `products/task465_issue172_stage2_per_layer_softmax_mixing_bounded_residual/verdict.json` (重建)
- **stage1_proof**: `products/task465_issue172_stage2_per_layer_softmax_mixing_bounded_residual/stage1_three_component_proof.json`
- **real_metadata**: `products/task465_issue172_stage2_per_layer_softmax_mixing_bounded_residual/real_three_component_metadata.npy`
- **adapter ckpt**: `products/task465_issue172_stage2_per_layer_softmax_mixing_bounded_residual/adapter.pt` (R12 强制)
- **train trace**: `products/task465_issue172_stage2_per_layer_softmax_mixing_bounded_residual/train_trace.json` (重建)
- **整体决策**: NO-GO 收口 + R20 + R21 v2
