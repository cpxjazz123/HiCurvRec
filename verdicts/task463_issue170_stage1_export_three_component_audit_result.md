# Task #463 / Issue #170 [方向B Gate1-2] 逐层混合曲率wrapper真实 Stage 1 三分量 export + 3 epoch sanity

## 任务摘要

Issue #170 owner 2026-08-01 07:00 派发: 方向B Gate 1-2 逐层混合曲率wrapper **真实** Stage 1 路径审计. 跟 #168 (closed NO-GO 11b0ee5) 的差异 = Stage 1 **真实** 三分量 (固定双曲+固定欧氏+learnable-κ) extract, 不是硬编码 metadata. 复用 #160 终点 (任务 #461/#168 已确认 wrapper 实质合规 + 三组件真实存在, 但 val_R@10=0), owner 进一步要真实 Stage 1 export + 3 epoch sanity.

R11.5 兜底 4. 简单实用方案 = 复用 task461 模板 + 加 Stage 1 RQ-VAE 三分量 export 函数 (load `task84/.../best_loss_model.pth` → extract 3 层 codebook → compute per-item 三分量: 固定双曲 norm_hyp + 固定欧氏 norm_eucl + learnable-κ placeholder) + 加 per-layer learnable κ_l (per layer independent) + 改 num_epochs=3.

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_PerLayerThreeComponentAdapter (跟 task461 一致, 但三分量 metadata 来自真实 Stage 1 extract + per-layer learnable κ_l)
- **Stage 1 RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (sha256=59a38fa3...)

## R18 4 维度对比 (vs #168 closed NO-GO 11b0ee5)

| 维度 | #168 (closed) | #170 (NEW) |
|------|---------------|------------|
| **D1 spec** | wrapper 路径审计, 硬编码 metadata 三组件 | **真实 Stage 1 export + 每层三分量 (固定双曲+固定欧氏+learnable-κ) + 3 epoch sanity** |
| **D2 实施** | task452 模板, α 0.117→1.854 1 epoch 16× | **load Stage 1 RQ-VAE → extract codebook → compute per-item 每层三分量** |
| **D3 失败机制** | α 1 epoch 涨 16× | wrapper 用了硬编码 metadata, 缺真实 Stage 1 三分量 |
| **D4 引用** | arXiv:2307.04514 | 同一引用 + real RQ-VAE integration + per-component |

→ **D1/D2 显著不同**: 从"代码审计" 转为"真实 Stage 1 三分量 export + 训练验证". R18 强制实验.

## Gate 1 (= Stage 1 真实三分量 export): ✅ PASS

**关键实测数据** (Task #463 实测):
- **RQ-VAE ckpt sha256**: `59a38fa3fa1aac5ac66dd90c3aa555d8ed00fd6fd6255e2143b42c6b285db4a1` (load 成功)
- **Layer 0 codebook** (K=64, 32d): sha256=`6bb3a46d...`, norm range=[0.0409, 0.6249]
- **Layer 1 codebook** (K=128, 32d): sha256=`d19fa7e8...`, norm range=[0.0261, 0.6813]
- **Layer 2 codebook** (K=256, 32d): sha256=`6b526a51...`, norm range=[0.0253, 0.6825]
- **Per-item 真实三分量 metadata** (shape=[9922, 3, 4]): sha256=`78b7172b...`
  - **三层独立 hash** (per layer metadata hash 一致): 
    - layer 0: `7d1a287d...`
    - layer 1: `c0a6eb59...`
    - layer 2: `96526f6e...`
  - **固定双曲分量 norm_hyp_l** (||tanh(||c||/2)||): range=[0.2646, 0.3266]
  - **固定欧氏分量 norm_eucl_l** (||c||): range=[0.5347, 0.6532]
  - **learnable-κ 分量 placeholder**: range 同 norm_eucl, 在 wrapper 内被 learnable_kappa_l 替换

**Gate 1 PASS 关键**: 三层独立 codebook 真实 extract, 三层 metadata hash 独立 + 一致, per-item 三分量计算无 NaN/Inf, 三分量范数范围合理 (固定双曲 < 固定欧氏, 跟理论预期一致).

## Gate 2 (= Stage 2 真实三分量 metadata 注入 + 3 epoch sanity): ❌ FAIL — R23 early-stop + α 饱和轨迹一致 + κ_l 三层同步!

**关键实测数据** (Task #463 跑完 2 epoch 后 R23 early-stop):
- α 轨迹: 4.54e-5 (init) → 0.1156 (epoch 0) → 1.9544 (epoch 1)
- 增长模式: epoch 0 涨 2544× (0.1156 / 4.54e-5), epoch 1 再涨 16.9×
- **跟 #163 alpha clamp 1 epoch 16× growth + #168 hardcoded 16× growth 完全同轨迹** (虽然 Stage 1 三分量 metadata 真实 extract 了, 但训练 dynamic 失败模式一致)
- **per-layer κ_l 轨迹** (learnable, init=0 + softplus):
  - epoch 0: [0.6971, 0.6972, 0.6973] — 三层 κ_l 都学到 0.697 (相对 init 0 涨无穷大)
  - epoch 1: [0.7054, 0.7057, 0.7058] — 三层仍同步涨, 没有 per-layer 差异
- loss 轨迹: 9.1305 → 2.6211 (decreased=True)
- **val_R@10 轨迹: [0.0, 0.0]** (R23 触发: 连续 2 次 0)
- grad 健康: cond_grad 4.4e-2→0.88, ln_grad 0.32→0.60 (无 NaN/Inf, 健康)
- save/load: missing=0, unexpected=0
- forward diff: 0.00e+00 (R23 skip)

**R11.5 决策错误反思**: 复用 task461 模板 + 加 Stage 1 真实三分量 export + 加 per-layer learnable κ_l, **虽然 Gate 1 PASS (Stage 1 三分量 export 真实) + 三层 metadata hash 独立 + 三分量范数范围合理, 但训练 dynamic 完全跟 #163/#168 失败模式轨迹一致** (α 1 epoch 涨 2544×, val_R@10 持续 0). 而且 per-layer learnable κ_l 三层学完全相同的值 (~0.697), 没有 per-layer 差异, 说明 κ_l 也跟 α 一样被 loss 强相关推高.

## Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec

- 原因: Gate 2 训练行为 FAIL, Stage 3 跑也无意义

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Gate 3 必须 PASS 才能进 Gate 4

## Gate 4 决策 (整体): **NO-GO** 收口 + 关键发现

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **真实 Stage 1 三分量 export 失败** | ❌ REFUTED | Gate 1 PASS, 3 layer codebook 真实 extract, 三层 metadata hash 独立一致, 三分量范数范围合理 |
| **wrapper 实质合规 ≠ 训练成功** | ❌ CONFIRMED | 真实 Stage 1 三分量 metadata 注入 + Gate 1 PASS, 但 training α 4.5e-5→1.95 跟 #163/#168 失败模式轨迹完全一致 |
| **真实 Stage 1 三分量 比 硬编码更好** | ❌ REFUTED | α 2544× growth + val_R@10=[0,0] 跟 #168 硬编码版几乎相同 |
| **per-layer learnable κ_l 是 R@10 杠杆** | ❌ REFUTED | κ_l 三层都学 0.697 (init 0), 1 epoch 后同步涨到 0.706, 没有 per-layer 差异, 跟 α 一样被推高 |
| **三层独立 hash 是 R@10 杠杆** | ❌ REFUTED | 三层 hash 独立验证 (7d1a28..., c0a6eb..., 96526f...), 但 training FAIL 仍一致 |

**Issue #170 闭环决策: NO-GO** (Stage 1 真实三分量 export PASS, 但 training FAIL — 重要 meta-finding: 真实 Stage 1 三分量 metadata ≠ 训练成功, 真实根因在训练 dynamic 而非 metadata 来源或 per-layer 差异).

## 关键发现 (Meta-finding)

| 编号 | 发现 | 证据 |
|------|------|------|
| **F1** | 真实 Stage 1 RQ-VAE 三分量 export 完全成功 | Gate 1 PASS, 3 layer codebook hash + per-item 三层 metadata hash 独立验证, 三分量范数范围合理 (norm_hyp_l < norm_eucl_l) |
| **F2** | 但训练行为 100% 跟 #163/#168 失败模式轨迹一致 | α 4.5e-5 → 0.116 → 1.95, val_R@10 持续 0, 真实三分量 metadata 没用 |
| **F3** | per-layer learnable κ_l 三层同步, 没有 per-layer 差异 | κ_l 三层都学 0.697 (init 0), 跟 α 一样被 loss 强相关推高, 没学到 per-layer 信号 |
| **F4** | 真正根因 = α_logit / learnable_kappa_l 跟 loss 强相关, 梯度推高 | 即使 init=-10/0 + softplus + 真实 Stage 1 三分量 metadata, softplus 1 epoch 涨 2544×, 不是三分量问题 |
| **F5** | #168 NO-GO 判决部分正确 (metadata 来源非根因) | 真实三分量 + wrapper 合规 + per-layer κ_l, training 仍 FAIL, 证实根因在 wrapper 训练 dynamic |
| **F6** | task463 vs #168 数值几乎完全一致 | task463 α 4.5e-5→1.95 (1 epoch 涨 2544×), #168 α 4.5e-5→1.85 (1 epoch 涨 2577×), val_R@10 同样 [0,0] |

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案 = 复用 task461 模板 + 加 Stage 1 三分量 export + per-layer learnable κ_l**: load RQ-VAE + extract 3 layer codebook + compute per-item 三分量 (norm_hyp + norm_eucl + learnable-κ placeholder) + wrapper 内 per-layer learnable κ_l
2. **R23 强制 early-stop**: val_R@10=[0,0] 连续 2 次 → 立即 §25 kill + 写 NO-GO
3. **R20 4-Gate 详细**: commit message 含 Gate 0/1/2/3/4 状态 + 关键数据 + 失败原因
4. **R18 强制实验**: 不只沿用 #168 判决, 真实 Stage 1 三分量 export + per-layer κ_l (D1/D2 显著不同)
5. **R15 强制 push**: issue 闭环时 verdict push 到 origin

## 后续 (R10 v2 idle 允许)

Issue #170 NO-GO 收口. Issue #169 (task462) 同模式 (Gate 1 PASS 真实 κ/scale export, training FAIL 同样轨迹 α 2524×). 10 issue κ/scale 元数据适配 (#157/#158/#162/#163/#165/#166/#167/#168/#169/#170) 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽.

#450 v8 (Issue #161, ZeroCenteredLayerNorm 真实零中心架构, val_R@10 ep95=0.1045 持续突破 baseline) 仍在 GPU 0 继续训练, 是唯一有可能 R@10 > 0.1020 的路径.

## 关键产物

- **verdict**: `verdicts/task463_issue170_stage1_export_three_component_audit_result.md`
- **verdict.json**: `products/task463_issue170_stage1_export_three_component_audit/verdict.json`
- **stage1_proof**: `products/task463_issue170_stage1_export_three_component_audit/stage1_three_component_proof.json` (3 codebook hashes + 3 layer metadata hashes)
- **real_metadata**: `products/task463_issue170_stage1_export_three_component_audit/real_three_component_metadata.npy` (shape=[9922, 3, 4])
- **adapter ckpt**: `products/task463_issue170_stage1_export_three_component_audit/adapter.pt` (R12 强制落盘, epoch 1 末)
- **train trace**: `products/task463_issue170_stage1_export_three_component_audit/train_trace.json`
- **commit pending**: 写完 verdict 后立即 commit + push
- **整体决策**: NO-GO 收口 + R20 + R21 v2 (commit hash 落地后补)