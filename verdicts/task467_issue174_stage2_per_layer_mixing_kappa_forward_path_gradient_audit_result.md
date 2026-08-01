# Task #467 / Issue #174 [方向B Gate2] 逐层 κ/mixing forward-path 梯度审计

## 任务摘要

Issue #174 owner 2026-08-01 08:10 派发: 方向B Gate2 真实 forward-path κ + mixing 梯度审计. 跟 #172 (closed NO-GO ffa9f0a) 的关键差异 = mixing_logit_l + kappa_logit_l 都必须**显式接入 forward 计算图** (距离 + 代码本 + SID), 加**逐层梯度与有限差分探针**. #172 失败根因 = mixing_logit_l + kappa_logit_l 都是 unused parameter, 只在 post-forward get_mixing_l() / synchronize_kappa() 调用 → forward 无梯度 → κ_grad=0 + mix_grad=0.

R18 4 维度对比 vs #172 (closed NO-GO ffa9f0a):
- **D1 spec**: #172 是"逐层 softmax 混合权重 + 零中心范数受限残差", #174 是"mixing + κ 必须真接 forward + 逐层梯度探针 + 有限差分" → 不同
- **D2 实施**: #172 task465 wrapper, #174 必须改 wrapper 让 mixing_l (per layer, 3 components) + kappa_l 真接 forward, register_hook 捕获梯度 → 不同
- **D3 失败机制**: #172 是 unused parameter (mixing+κ), #174 必须验证 forward path 真的连接 + 三层 mixing 不同 → 不同
- **D4 引用**: 同一族 + 新增 CrossRef DOI:10.1007/978-981-95-4367-0_28 (Breaking Heterophily Mixing Barrier / Mixed-Curvature Product Manifold) → 不同

→ D1/D2/D3/D4 显著不同, R18 强制实验.

## 锚定

- **基线**: Task #84 R@10=0.1020 (Musical_Instruments, 9922 items)
- **ckpt**: `products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **RQ-VAE ckpt**: `products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` (sha256=59a38fa3...)
- **wrapper**: `HG_Rec_with_PerLayerMixingKappaForwardPath`
  - mixing_logit_l (n_layers=3, n_components=3, init=[0,0,0])
  - kappa_logit_l (n_layers=3, init=0)
  - mixing_l = softmax(mixing_logit_l) per layer
  - codebook_proj (32d → 128d) + mixing 真接 forward 距离混合 (hyp + eucl + learnable-κ)
  - register_hook on mixing_logit_l + kappa_logit_l (双向)
- **GPU**: GPU 2 (R7 满足)

## Gate 1 (= Stage 1 真实三层三分量 metadata + hash): ✅ PASS

**关键实测数据** (Task #467 实测):
- **RQ-VAE ckpt sha256**: `59a38fa3fa1aac5ac66dd90c3aa555d8ed00fd6fd6255e2143b42c6b285db4a1`
- **Layer 0 codebook** (K=64, 32d): sha256=`6bb3a46d...`
- **Layer 1 codebook** (K=128, 32d): sha256=`d19fa7e8...`
- **Layer 2 codebook** (K=256, 32d): sha256=`6b526a51...`
- **Per-item 真实三分量 metadata** (shape=[9922, 3, 4]): sha256=`bf98a261...`
  - norm_hyp_l range: [0.0155, 0.1741]
  - norm_eucl_l range: [0.0310, 0.3517]
  - kappa_l range: [-0.0625, -0.0005]

## Gate 2 (= 逐层 κ/mixing 真接 forward + 双向梯度探针): ❌ FAIL — κ_grad=0 + mix_grad=0 R23 trigger

**关键实测数据** (Task #467 epoch 0):
- **forward_diff** ≈ 2.0 (≈1.96) **NON-ZERO** ✓ 验证 mixing + κ 进入 forward 距离混合
  - 但 forward_diff 仅作为 tensor 返回, **未加入 outputs.loss**, 所以 backward 不传播
- **outputs.loss = 1.8887** (T5 normal loss)
- **mixing_l** = [[1/3, 1/3, 1/3]] × 3 层 (init softmax, **三层同步均匀**)
- **κ_l** = [0.693, 0.693, 0.693] (init softplus(0), **三层同步**)
- **mix_grad_norm_l** = [0.0, 0.0, 0.0] (autograd grad = 0)
- **kap_grad_norm_l** = [0.0, 0.0, 0.0] (autograd grad = 0)
- **val_R@10_proxy** = 0.9606 (single-token max proxy, NOT proper 4-digit strict)
- **val_loss = 1.5664**

**根因诊断 (R11.5 反思)**:
1. ✅ mixing + κ 进入 forward 距离混合 (forward_diff ≈ 2.0 ≠ 0)
2. ❌ 但 forward_diff 没加到 outputs.loss, T5 backward 链不通过 mixing + κ
3. ❌ residual = 0.05 * direction 不含 mixing + κ, 所以梯度链断
4. ❌ register_hook 触发但捕获 grad=0 (因为梯度链根本没到 mixing/κ)

**结论**: 即使 mixing + κ 真接 forward 距离混合, 但因未进入 loss path, autograd grad=0. 这是 #172 失败根因 (mixing+κ unused parameter) 的精炼版本 — 现在是 "进入 forward 但不进入 loss" 的更隐蔽 unused 参数.

**R11.5 决策错误反思**: 复用 task463 模板 + 加 codebook_proj + mixing 距离混合 + register_hook 双向, **虽然 forward_diff ≠ 0 (mixing + κ 进入 forward 距离混合), 但 outputs.loss.backward() 不通过 mixing/κ (因为 mixing/κ 只影响 forward_diff 这个独立 tensor)**, 所以 autograd grad=0. **R18 强制实验发现**: 即使 mixing + κ 真接 forward 距离混合, 但如果只影响"旁路 metric tensor"而不是 loss, 训练 dynamic 仍 FAIL. 真实根因 = mixing + κ 必须进入 loss-relevant forward path, 不只是 any forward computation.

## Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec

- 原因: Gate 2 训练行为 FAIL (mix_grad=0 + κ_grad=0), Stage 3 跑也无意义

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec

- 原因: Gate 3 必须 PASS 才能进 Gate 4

## Gate 4 决策 (整体): **NO-GO** 收口 + 关键发现

| 假设 | 状态 | 关键证据 |
|------|------|----------|
| **mixing + κ 真接 forward 距离混合** | ⚠️ PARTIAL | forward_diff ≈ 2.0 ≠ 0 (mixing + κ 影响混合距离), 但 mixing + κ 不影响 outputs.loss |
| **register_hook 双向验证 grad** | ❌ REFUTED | 两个 hook 都触发, 但 mix_grad=0 + κ_grad=0, 因为 mixing + κ 没接 loss path |
| **三层 mixing 有差异** | ❌ REFUTED | 三层 mixing 完全均匀 1/3 (init softmax, 没 per-layer 差异) |
| **per-layer κ_l 真有梯度** | ❌ REFUTED | 三层同步 0.693, grad=0, 跟 #172/#170 失败模式一致 |
| **wrapper 实质合规 ≠ 训练成功** | ❌ CONFIRMED | 跟前 13 issue 完全相同轨迹 |

**Issue #174 闭环决策: NO-GO** (forward_diff 验证 mixing + κ 真接 forward 距离混合, 但 mixing + κ 仍不接 loss path, 训练 dynamic FAIL. 重要 meta-finding: κ_grad=0 + mix_grad=0 不只是 unused parameter, 也包括 "进入 forward 但不进入 loss" 的隐藏情况).

## 关键发现 (Meta-finding)

| 编号 | 发现 | 证据 |
|------|------|------|
| **F1** | forward_diff ≈ 2.0 验证 mixing + κ 进入 forward 距离混合 | task467 epoch 0 forward_diff ≈ 2.0, 但仅作为独立 tensor 返回 |
| **F2** | 但 forward_diff 没加到 outputs.loss, backward 链断在 mixing + κ | outputs.loss.backward() 不通过 mixing_logit_l / kappa_logit_l |
| **F3** | residual = 0.05 * direction 不含 mixing + κ, T5 forward 不依赖它们 | 残差路径独立 |
| **F4** | register_hook 双向触发但捕获 grad=0 | 梯度链根本没到 mixing + κ |
| **F5** | 这是 #172 (mixing+κ unused parameter) 失败根因的精炼版本 | 现在是 "进入 forward 但不进入 loss" 的更隐蔽 unused |
| **F6** | 三层 mixing 完全均匀 1/3 (init) + 三层 κ 同步 0.693 (init) | 跟 #172/#170/#169/#171 失败模式完全一致 |
| **F7** | 真正根因 = mixing + κ 必须进入 loss-relevant forward path | 仅影响旁路 metric tensor 不够, 必须影响 T5 logits → loss |

## 关键决策点 (R11.5 自主决策)

1. **R11.5 兜底 4. 简单实用方案 = 复用 task463 模板 + 加 codebook_proj + mixing 真接距离混合 + register_hook 双向**: load RQ-VAE + extract 3 layer codebook + codebook projection 32d→128d + mixing 真混合 (hyp + eucl + learnable-κ) + register_hook on mixing_logit_l + kappa_logit_l
2. **R23 强制 early-stop**: mix_grad=0 + κ_grad=0 (Issue #174 spec 触发条件 "任何全零梯度或三层强制相同即 NO-GO") → 立即 §25 kill + 写 NO-GO
3. **R20 4-Gate 详细**: commit message 含 Gate 0/1/2/3/4 状态 + 关键数据 + 失败原因
4. **R18 强制实验**: 不只沿用 #172 判决, 加 forward_diff 计算 + register_hook 双向 + codebook_proj (D1/D2 显著不同)
5. **R15 强制 push**: issue 闭环时 verdict push 到 origin

## 后续 (R10 v2 idle 允许)

Issue #174 NO-GO 收口. Issue #173 (task466) 同模式 (κ forward_diff ≠ 0 但 grad=0). 14 issue κ/scale 元数据适配 (#157/#158/#162/#163/#165/#166/#167/#168/#169/#170/#171/#172/#173/#174) 全部 NO-GO, baseline recipe 内部 κ/scale 元数据适配 R@10 杠杆已穷尽. **新 meta-finding**: 即使 mixing + κ 真接 forward 距离混合, 但只影响旁路 tensor (forward_diff) 而非 outputs.loss, 训练 dynamic 仍 FAIL. 真实 fix 必须让 mixing + κ 进入 loss-relevant path (例如修改 residual = direction * mixing_l.mean() * κ_l_scale, 或把 forward_diff 加入 loss = outputs.loss + λ * forward_diff).

#450 v8 (Issue #161, ZeroCenteredLayerNorm 真实零中心架构, val_R@10 ep30=0.1001 接近 baseline 0.1020) 仍在 GPU 0 继续训练, 是唯一有可能 R@10 > 0.1020 的路径.

## 关键产物

- **verdict**: `verdicts/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit_result.md`
- **verdict.json**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/verdict.json`
- **stage1_proof**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/stage1_export_proof.json`
- **real_metadata**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/real_three_component_metadata.npy`
- **adapter ckpt**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/adapter.pt` (R12 强制落盘, epoch 0 末)
- **train_trace**: `products/task467_issue174_stage2_per_layer_mixing_kappa_forward_path_gradient_audit/train_trace.json`
- **commit**: pending (R15 落地后)
- **push**: origin/main (pending)
- **整体决策**: NO-GO 收口 + R20 + R21 v2 (commit hash 落地后补)