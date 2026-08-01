# Task #452 / Issue #160 Gate3 [方向B] 三分量加权混合曲率元数据 T5 逐层几何适配 — 短训 Gate 3 PASS

## 关键结论

**Gate 3: ✅ PASS** (per Issue #160 spec)

## Gate 1 (= Stage 1 RQ-VAE): ⏸ STOP per spec

- 原因: Issue #160 spec 仅要求 Gate 3 Stage 3, 不要求 Stage 1 RQ-VAE 重训
- Issue spec 强制: Gate 1/2 沿用 Issue #158 Gate 2 PASS (commit fa0b455) 的三分量加权混合曲率 SID 链路 (Task #449)
- precheck 5/5 PASS: α=0 → max_diff≈0 (init identity, 软上限 1e-3, 实际 1.19e-05); 仅 LN + conditioner 解冻 (trainable=10, frozen=104); 首步后 conditioner + LN 梯度 nonzero; SID range 全部 [0, 9922); SID SHA256 hash 跟 #158 Gate 2 一致

## Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec

- 原因: Issue #160 spec 不要求 Stage 2 Sinkhorn 重跑, 沿用 Issue #158 Gate 2 PASS 的 SID 链路
- Issue spec 强制: Gate 2 目标 = 三分量加权混合曲率后 4-digit SID 唯一性 ≥ 9500/9922, 已由 #158 闭环验证

## Gate 3 (= Stage 3 T5-mini): ✅ PASS

### 关键数据

- **epoch0 loss=9.1448, epoch9 loss=6.68e-05** (-100.0%, 收敛)
- **cond_grad trace**: ['4.53e-02', '1.01e+00', '1.06e-01', '2.65e-02', '9.08e-03', '2.91e-03', '1.04e-03', '5.29e-04', '4.32e-04', '3.42e-04'] (持续衰减, 全 finite)
- **ln_grad trace**: ['2.97e-01', '5.33e-01', '3.26e-02', '7.14e-03', '2.50e-03', '9.29e-04', '3.43e-04', '1.52e-04', '1.13e-04', '8.42e-05'] (LN 梯度全程稳定, 跟 task451 #159 同样模式)
- **α trace**: 0.12 → 1.85 → 2.32 → 2.98 → 4.02 → 5.45 → 7.17 → 9.46 → 12.32 → **15.41** (单调递增, 跟 #159 同 softplus unbounded 设计)
- **nan_inf 全程 False**
- **adapter.pt SHA256**: `34abfc4f0a926e163c71246f1288f8ebbc18e4f7434176c66396b1cc7793ab78` (R12 ckpt 强制)
- **save/load missing=0/unexpected=0**, **forward diff max = 0.00e+00** (跨 reload 完全一致)

### 失败原因

无 (Gate 3 PASS, 7/7 check 全通过: loss 收敛 + cond_grad nonzero 全程 + ln_grad nonzero 全程 + nan_inf False + sid_range True + save/load 一致 + forward diff=0)

### 实施

- 脚本: `scripts/task452_issue160_gate3_weighted_mixed_curvature_meta.py` (594 行, R4 py_compile PASS)
- adapter: `WeightedMixedCurvatureConditioner` (curvature_embed 12 维 per sample: 3 layers × 4 dims [κ_l, alpha_l, beta_l, gamma_l])
- wrapper: `HG_Rec_with_WeightedMixedAdapter` 走 `self.t5.model(inputs_embeds=...)` 注入零中心几何残差
- trainable: input LayerNorm + 三分量 conditioner (其余 T5 全冻结)
- 训练: 10 epoch short-train (~12 min on L40S GPU 2)
- Issue spec 强制: 5 个连续 epoch 梯度 < 阈值 → FAIL, 实际未触发

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Issue #160 spec 仅要求 Gate 3 10 epoch 短训, Gate 4 200 epoch + Stage 4 双复跑待 owner 决策 (R11.4 critical: GPU 几小时)
- Issue spec 强制: Gate 4 R@10 阈值 vs HG-Rec baseline 0.1020, GO/NO-GO 判定在 Stage 4 跑通后给出

## α 暴涨机制说明 (R11.5 自主决策)

α 从 0.12 → 15.41 单调递增, 跟 Issue #159 κ-scale 同模式. 三分量加权混合曲率提供比通用零中心残差 (Issue #150) 更强的 gradient signal, α 增长是设计外但 spec 未限. 当前 Gate 3 PASS 但 α 增长斜率 (+1.7/epoch) 暗示 200 epoch 时 α 会达 e^40 量级, **Stage 4 启动前需要 owner 决策是否加 α bound** (R11.5 critical, 跟 #159 同步).

## 关键产物

- verdict: `verdicts/task452_issue160_gate3_weighted_mixed_curvature_meta_result.md` (本文件)
- ckpt: `products/task452_issue160_gate3_weighted_mixed_curvature_meta/adapter.pt` (404421 bytes)
- 训练 trace: `products/task452_issue160_gate3_weighted_mixed_curvature_meta/train_trace.json`
- 8 件套全落盘: config.json + adapter_init_proof.json + layernorm_unfreeze_proof.json + gradient_proof.json + train_trace.json + verdict.json + adapter.pt + log
- 整体决策: **Gate 3 ✅ PASS, Gate 4 ⏸ STOP per spec 等 owner Stage 4 启动决策**