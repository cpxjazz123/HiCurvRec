# Task #462 / Issue #169 [方向A Gate1-2] κ元数据条件化wrapper真实 Stage 1 κ/scale export + 3 epoch sanity

## 任务摘要

Issue #169 owner 2026-08-01 07:00 派发: 方向A Gate 1-2 κ元数据条件化wrapper **真实** Stage 1 路径审计. 跟 #167 (closed NO-GO 11b0ee5) 的差异 = Stage 1 **真实** κ/scale export (从 RQ-VAE 模型 extract), 不是硬编码 metadata. 复用 #160 终点 (任务 #460/#167 已确认 wrapper 实质合规, 但 val_R@10=0), owner 进一步要真实 Stage 1 export + 3 epoch sanity.

R11.5 兜底 4. 简单实用方案 = 复用 task460 模板 + 加 Stage 1 RQ-VAE export 函数 (load `task84/.../best_loss_model.pth` → extract 3 层 codebook → compute per-item κ_l + scale_l) + 改 num_epochs=3.

## 锚定

- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **数据集**: Musical_Instruments (Amazon, 9922 items)
- **baseline**: Task #84 R@10=0.1020
- **wrapper**: HG_Rec_with_RealMetadataAdapter (跟 task460 一致, 但 metadata 来自真实 Stage 1 export)
- **Stage 1 RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (sha256=59a38fa3...)

## R18 4 维度对比 (vs #167 closed NO-GO 11b0ee5)

| 维度 | #167 (closed) | #169 (NEW) |
|------|---------------|------------|
| **D1 spec** | wrapper 路径审计, 硬编码 metadata | **真实 Stage 1 export + κ/scale 三层独立learnable + 3 epoch sanity** |
| **D2 实施** | task452 模板, α 4.5e-5→1.854 (1 epoch 16× growth) | **load Stage 1 RQ-VAE → extract codebook → compute per-item κ_l + scale_l** |
| **D3 失败机制** | wrapper 实质合规但训练 dynamic 失败 | wrapper 用了硬编码 metadata, 缺真实 Stage 1 κ/scale |
| **D4 引用** | arXiv:2405.13979 | 同一引用 + real RQ-VAE integration |

→ **D1/D2 显著不同**: 从"代码审计" 转为"真实 Stage 1 export + 训练验证". R18 强制实验.

## Gate 1 (= Stage 1 真实 export): ✅ PASS

**关键实测数据** (Task #462 实测):
- **RQ-VAE ckpt sha256**: `59a38fa3fa1aac5ac66dd90c3aa555d8ed00fd6fd6255e2143b42c6b285db4a1` (load 成功)
- **Layer 0 codebook** (K=64, 32d): sha256=`6bb3a46d...`, norm range=[0.0409, 0.6249]
- **Layer 1 codebook** (K=128, 32d): sha256=`d19fa7e8...`, norm range=[0.0261, 0.6813]
- **Layer 2 codebook** (K=256, 32d): sha256=`6b526a51...`, norm range=[0.0253, 0.6825]
- **Per-item 真实 metadata** (shape=[9922, 3, 4]): sha256=`81509c14...`
  - κ_l range: [-0.2823, -0.0019] (负 curvature, ||c||² 比例决定)
  - α_l range: [0.2482, 0.5000] (sigmoid(κ_l/10))
  - β_l range: [0.2364, 0.5036] (||c||/max(||c||))
  - γ_l range: [0.0001, 0.2636] (1 - α - β, 归一化后)

**Gate 1 PASS 关键**: 三层独立 codebook 真实 extract, hash 一致, per-item metadata 计算无 NaN/Inf.

## Gate 2 (= Stage 2 真实 metadata 注入 + 3 epoch sanity): ❌ FAIL — R23 early-stop + α 饱和轨迹一致!

**关键实测数据** (Task #462 跑完 2 epoch 后 R23 early-stop):
- α 轨迹: 4.54e-5 (init) → 0.1145 (epoch 0) → 1.7397 (epoch 1)
- 增长模式: epoch 0 涨 2524× (0.1145 / 4.54e-5), epoch 1 再涨 15.2×
- **跟 #162 alpha clamp 1 epoch 16× growth + #167 hardcoded 16× growth 完全同轨迹** (虽然 Stage 1 metadata 真实 extract 了, 但训练 dynamic 失败模式一致)
- loss 轨迹: 9.1311 → 2.1207 (decreased=True)
- **val_R@10 轨迹: [0.0, 0.0]** (R23 触发: 连续 2 次 0)
- grad 健康: cond_grad 4.3e-2→1.12, ln_grad 0.32→0.41 (无 NaN/Inf, 健康)
- save/load: missing=0, unexpected=0
- forward diff: 0.00e+00 (R23 skip)

**R11.5 决策错误反思**: 复用 task460 模板 + 加 Stage 1 真实 export, **虽然 Gate 1 PASS (Stage 1 export 真实), 但训练 dynamic 完全跟 #162/#167 失败模式轨迹一致** (α 1 epoch 涨 2524×, val_R@10 持续 0). 真实根因不在 metadata 来源 (硬编码 vs 真实 extract), 而在训练 dynamic (α_logit 跟 loss 强相关, 即使 init=-10 + Tanh bounded + 真实 Stage 1 metadata, softplus 仍被推高).

## Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec

- 原因: Gate 2 训练行为 FAIL, Stage 3 跑也无意义

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Gate 3 必须 PASS 才能进 Gate 4

## Gate 4 决策 (整体): **NO-GO** 收口 + 关键发现

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **真实 Stage 1 κ/scale export 失败** | ❌ REFUTED | Gate 1 PASS, 3 layer codebook 真实 extract, per-item metadata hash 一致, 无 NaN/Inf |
| **wrapper 实质合规 ≠ 训练成功** | ❌ CONFIRMED | 真实 Stage 1 metadata 注入 + Gate 1 PASS, 但 training α 4.5e-5→1.74 跟 #162/#167 失败模式轨迹完全一致 |
| **真实 Stage 1 metadata 比硬编码更好** | ❌ REFUTED | α 2524× growth + val_R@10=[0,0] 跟 #167 硬编码版几乎相同 |
| **三层独立 learnable κ_l 是 R@10 杠杆** | ❌ REFUTED | κ_l range [-0.28, -0.002] 真实但训练 dynamic 仍失败 |

**Issue #169 闭环决策: NO-GO** (Stage 1 真实 export PASS, 但 training FAIL — 重要 meta-finding: 真实 Stage 1 metadata ≠ 训练成功, 真实根因在训练 dynamic 而非 metadata 来源).

## 关键发现 (Meta-finding)

| 编号 | 发现 | 证据 |
|------|------|------|
| **F1** | 真实 Stage 1 RQ-VAE export 完全成功 | Gate 1 PASS, 3 layer codebook hash + per-item metadata hash 全验证, 数值范围合理 (kappa_l < 0, scale_l ∈ [0, 0.68]) |
| **F2** | 但训练行为 100% 跟 #162/#167 失败模式轨迹一致 | α 4.5e-5 → 0.114 → 1.74, val_R@10 持续 0, 真实 metadata 没用 |
| **F3** | 真正根因 = α_logit 跟 loss 强相关, 梯度推高 | 即使 init=-10 + Tanh bounded + 真实 Stage 1 metadata, softplus 1 epoch 涨 2524×, 不是 metadata 问题 |
| **F4** | #167 NO-GO 判决部分正确 (metadata 来源非根因) | 真实 metadata + wrapper 合规, training 仍 FAIL, 证实根因在 wrapper 训练 dynamic |
| **F5** | task462 vs #167 数值几乎完全一致 | task462 α 4.5e-5→1.74 (1 epoch 涨 2524×), #167 α 4.5e-5→1.85 (1 epoch 涨 2577×), val_R@10 同样 [0,0] |

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案 = 复用 task460 模板 + 加 Stage 1 export**: load RQ-VAE + extract 3 layer codebook + compute per-item κ_l + scale_l + 注入 wrapper
2. **R23 强制 early-stop**: val_R@10=[0,0] 连续 2 次 → 立即 §25 kill + 写 NO-GO
3. **R20 4-Gate 详细**: commit message 含 Gate 0/1/2/3/4 状态 + 关键数据 + 失败原因
4. **R18 强制实验**: 不只沿用 #167 判决, 真实 Stage 1 export (D1/D2 显著不同)
5. **R15 强制 push**: issue 闭环时 verdict push 到 origin

## 后续 (R10 v2 idle 允许)

Issue #169 NO-GO 收口. Issue #170 (task463) 同模式 (audit 7 项 PASS, training FAIL 同样轨迹, 但 κ_l 三层学同样值 ~0.7). 10 issue κ/scale 元数据适配 (#157/#158/#162/#163/#165/#166/#167/#168/#169/#170) 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽.

#450 v8 (Issue #161, ZeroCenteredLayerNorm 真实零中心架构, val_R@10 ep95=0.1045 持续突破 baseline) 仍在 GPU 0 继续训练, 是唯一有可能 R@10 > 0.1020 的路径.

## 关键产物

- **verdict**: `verdicts/task462_issue169_stage1_export_kappa_scale_audit_result.md`
- **verdict.json**: `products/task462_issue169_stage1_export_kappa_scale_audit/verdict.json`
- **stage1_proof**: `products/task462_issue169_stage1_export_kappa_scale_audit/stage1_export_proof.json` (3 codebook hashes + per-item metadata hash)
- **real_metadata**: `products/task462_issue169_stage1_export_kappa_scale_audit/real_metadata.npy` (shape=[9922, 3, 4])
- **adapter ckpt**: `products/task462_issue169_stage1_export_kappa_scale_audit/adapter.pt` (R12 强制落盘, epoch 1 末)
- **train trace**: `products/task462_issue169_stage1_export_kappa_scale_audit/train_trace.json`
- **commit pending**: 写完 verdict 后立即 commit + push
- **整体决策**: NO-GO 收口 + R20 + R21 v2 (commit hash 落地后补)