## Issue #159 R20+R21 v2 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 RQ-VAE): ⏸ STOP per spec
- 关键数据: Issue #159 spec 仅要求 Gate 3 Stage 3, 不要求 Stage 1 RQ-VAE 重训
- 失败原因: 不适用 (per spec)
- 实施: 沿用 Issue #157 Gate 2 PASS (commit 206ebb5) 的 κ 同步重校准 SID 链路 (Task #448)
- precheck 5/5 PASS: α=0 → max_diff=1.81e-5; 仅 LN + conditioner 解冻 (trainable=12, frozen=104); 首步 conditioner + LN 梯度 nonzero; SID SHA256 hash 跟 #157 Gate 2 一致

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Issue #159 spec 不要求 Stage 2 Sinkhorn 重跑
- 实施: 沿用 Issue #157 Gate 2 PASS 的 SID 链路 (4-digit SID 唯一性 ≥ 9500/9922)

### Gate 3 (= Stage 3 T5-mini): ✅ PASS (per Issue #159 spec)
- 关键数据: epoch0 loss=9.1460 → epoch9 loss=0.0006 (-99.99% 收敛); cond_grad trace ['4.40e-02', '8.46e-01', '5.93e-02', '8.83e-03', '1.98e-02', '2.30e-03', '8.24e-04', '1.38e-03', '5.77e-04', '5.15e-03'] 全 finite; ln_grad trace ['2.97e-01', '7.28e-01', '1.40e-01', '2.80e-02', '8.91e-03', '3.47e-03', '1.84e-03', '1.15e-03', '7.64e-04', '4.90e-04'] 持续稳定; α 0.12→18.75 (softplus unbounded, 设计外但 spec 未限); nan_inf 全程 False
- 失败原因: 无 (Gate 3 PASS, 7/7 check 全通过)
- 实施: `scripts/task451_issue159_gate3_kappa_scale_meta.py` (575 行, R4 py_compile PASS); `KappaScaleConditioner` (kappa_embed + scale_embed + conditioner MLP); `HG_Rec_with_KappaScaleAdapter` 走 `self.t5.model(inputs_embeds=...)` 注入零中心几何残差; trainable: input LayerNorm + κ-scale conditioner; 10 epoch short-train ~12 min on L40S GPU 1
- ckpt SHA256: `3d79f3875c94a276a8aa5c562cee7eac53c022d2415471d07bf40693f7c160cc` (R12 ckpt 强制)
- verdict 路径: `verdicts/task451_issue159_gate3_kappa_scale_meta_result.md`

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Issue #159 spec 仅要求 Gate 3 10 epoch 短训, Gate 4 200 epoch + 双复跑待 owner 决策 (R11.4 critical)
- 实施: Gate 4 R@10 阈值 vs HG-Rec baseline 0.1020, GO/NO-GO 判定在 Stage 4 跑通后给出; **α 增长斜率 (+2/epoch) 暗示 200 epoch 时 α 会达 e^50 量级, Stage 4 启动前 owner 决策是否加 α bound** (R11.5 critical)

### 关键产物
- commit hash: `66210fa`
- push: origin/main
- verdict: `verdicts/task451_issue159_gate3_kappa_scale_meta_result.md`
- ckpt: `products/task451_issue159_gate3_kappa_scale_meta/adapter.pt`
- 训练 trace: `products/task451_issue159_gate3_kappa_scale_meta/train_trace.json`
- 整体决策: **Gate 3 ✅ PASS, Gate 4 ⏸ STOP per spec 等 owner Stage 4 启动决策**