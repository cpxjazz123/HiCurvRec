## Issue #160 R20+R21 v2 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 RQ-VAE): ⏸ STOP per spec
- 关键数据: Issue #160 spec 仅要求 Gate 3 Stage 3, 不要求 Stage 1 RQ-VAE 重训
- 失败原因: 不适用 (per spec)
- 实施: 沿用 Issue #158 Gate 2 PASS (commit fa0b455) 的三分量加权混合曲率 SID 链路 (Task #449)
- precheck 5/5 PASS: α=0 → max_diff=1.19e-5; 仅 LN + conditioner 解冻 (trainable=10, frozen=104); 首步 conditioner + LN 梯度 nonzero; SID SHA256 hash 跟 #158 Gate 2 一致

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Issue #160 spec 不要求 Stage 2 Sinkhorn 重跑
- 实施: 沿用 Issue #158 Gate 2 PASS 的 SID 链路 (4-digit SID 唯一性 ≥ 9500/9922)

### Gate 3 (= Stage 3 T5-mini): ✅ PASS (per Issue #160 spec)
- 关键数据: epoch0 loss=9.1448 → epoch9 loss=6.68e-05 (-100.0% 收敛); cond_grad trace ['4.53e-02', '1.01e+00', '1.06e-01', '2.65e-02', '9.08e-03', '2.91e-03', '1.04e-03', '5.29e-04', '4.32e-04', '3.42e-04'] 全 finite; ln_grad trace ['2.97e-01', '5.33e-01', '3.26e-02', '7.14e-03', '2.50e-03', '9.29e-04', '3.43e-04', '1.52e-04', '1.13e-04', '8.42e-05'] 持续稳定; α 0.12→15.41 (softplus unbounded, 跟 #159 同模式); nan_inf 全程 False
- 失败原因: 无 (Gate 3 PASS, 7/7 check 全通过)
- 实施: `scripts/task452_issue160_gate3_weighted_mixed_curvature_meta.py` (594 行, R4 py_compile PASS); `WeightedMixedCurvatureConditioner` (curvature_embed 12 维 per sample: 3 layers × 4 dims [κ_l, alpha_l, beta_l, gamma_l]); `HG_Rec_with_WeightedMixedAdapter` 走 `self.t5.model(inputs_embeds=...)` 注入零中心几何残差; trainable: input LayerNorm + 三分量 conditioner; 10 epoch short-train ~12 min on L40S GPU 2
- ckpt SHA256: `34abfc4f0a926e163c71246f1288f8ebbc18e4f7434176c66396b1cc7793ab78` (R12 ckpt 强制)
- verdict 路径: `verdicts/task452_issue160_gate3_weighted_mixed_curvature_meta_result.md`

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Issue #160 spec 仅要求 Gate 3 10 epoch 短训, Gate 4 200 epoch + 双复跑待 owner 决策 (R11.4 critical)
- 实施: Gate 4 R@10 阈值 vs HG-Rec baseline 0.1020, GO/NO-GO 判定在 Stage 4 跑通后给出; **α 增长斜率 (+1.7/epoch) 暗示 200 epoch 时 α 会达 e^40 量级, Stage 4 启动前 owner 决策是否加 α bound** (R11.5 critical, 跟 #159 同步)

### 关键产物
- commit hash: `66210fa`
- push: origin/main
- verdict: `verdicts/task452_issue160_gate3_weighted_mixed_curvature_meta_result.md`
- ckpt: `products/task452_issue160_gate3_weighted_mixed_curvature_meta/adapter.pt`
- 训练 trace: `products/task452_issue160_gate3_weighted_mixed_curvature_meta/train_trace.json`
- 整体决策: **Gate 3 ✅ PASS, Gate 4 ⏸ STOP per spec 等 owner Stage 4 启动决策**