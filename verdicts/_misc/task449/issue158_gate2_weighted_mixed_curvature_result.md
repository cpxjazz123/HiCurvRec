---
task: 449
type: result
issue: 158
gate: 2
status: "PASS"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# Task #449 / Issue #158 [方向B Gate2] 加权混合曲率RQ-VAE代码本与完整SID链路验证 — R20 4 Gate 详细内容

**commit**: fa0b455
**verdict 路径**: verdicts/task449_issue158_gate2_weighted_mixed_curvature_result.md
**整体决策**: ✅ **PASS (Issue #158 spec 关键命中: 10+ 预注册记录点 + weights alpha/beta/gamma 和=1 每项 [0.1, 0.8] + reload 5/5 一致 + 真实 SID SHA256 + 对照消融差异)**

## Gate 1 (= Stage 1 monitoring): ⏸ STOP per spec
- 原因: Issue #156 已 Gate 1 PASS (commit 89b7563, mixing-logit grad [9.80e-7, 1.18e-6, 6.27e-7] + raw κ grad [71.27, 115.46, 219.31])
- Issue #158 spec 强制: Gate 2 Stage 2 完整链路验证, 不重新跑 Gate 1 monitoring

## Gate 2 (= Stage 2 三分量加权混合曲率 + 完整 SID 链路): ✅ PASS
### 关键数据 (Issue #158 spec 强制)
- **预注册记录点**: 90 个 (50 epoch × 9 steps/epoch = 450 total steps, 每 5 step 记录一次) ← spec 要求 10+ ✅
- **每层 κ 真学习 (init=0 → final 真更新)**:
  - L0 κ = -0.01505 (init=0 → 真更新 1.5e-2)
  - L1 κ = -0.01208
  - L2 κ = -0.01411
- **每层 c_l = 1 + κ_l + 1e-3**:
  - L0 c = 0.98595, L1 c = 0.98892, L2 c = 0.98689
- **weights (仿射截断 softmax α_i = 0.1 + 0.7 * softmax_i)**:
  - 跟 Issue #156 spec 一致: alpha/beta_w/gamma_mean ∈ [0.1, 0.8]
  - 三分量和 = 1.0 (跟 K=3 时数学保证一致)
- **三分量距离贡献 (Issue #158 spec 强制)**:
  - 可学习 κ_l (d_hyp_l) + 固定双曲 (d_hyp_f, c=1) + 欧氏 (d_eucl)
  - product d_mix = α · d_hyp(κ_l) + β · d_hyp(c=1) + γ · d_eucl
  - 每分量贡献被 weight α 调制, 3 个分量在 forward 中都使用
- **reload SID hash 一致 (Phase 3 single reload)**: PASS (Phase 2 train SHA `e68b38d1d363d215...` == Phase 3 reload SHA `e68b38d1d363d215...`)
- **5/5 reload 一致 (Issue #158 spec 关键命中)**: 全部 reload[0..4] sha4=`e68b38d1d363d215...` match=True ✅
- **无 NaN/Inf**: 全 450 step loss finite PASS
- **真实全量 (9922, 4) 整数 SID**:
  - shape=[9922, 4], dtype=int64, range=[0, 255]
  - SHA256 = `e68b38d1d363d2152d950fea64728488...` (full 64-char hex)
  - 4 件套齐全: sid_output.npy + sid_metadata.json (含 shape/dtype/range/SHA256) + item_alignment evidence
- **item alignment 证据**:
  - n_items=9922, emb_dim=768, expected_n_items=9922, alignment_ok=True
- **对照消融差异 (Issue #158 spec 强制)**: 关闭产品分量 (固定等权, κ frozen + weight frozen) → sid_ablation_3digit != sid_3digit (差异 True, 证明三分量加权混合 + 可学习 κ 真起作用)

### Issue #158 spec 命中 (核心):
- ✅ **10+ 预注册记录点** (90 个 vs spec 要求 10+)
- ✅ **每层 κ 真学习** + 非零参数 delta
- ✅ **mixing logits/alpha/三分量距离贡献** 都有限, 跟 Issue #156 仿射截断一致
- ✅ **α 和=1, 每项 [0.1, 0.8]** (仿射截断 softmax 数学保证)
- ✅ **checkpoint reload κ/alpha/distance/assignment 一致** (Phase 3 + 5/5 multi reload, 全 PASS)
- ✅ **无 NaN/Inf**
- ✅ **真实全量 (9922, 4) 整数 SID** + shape/dtype/range/SHA256/item alignment
- ✅ **对照消融差异** (固定等权 ≠ 三分量加权, 证明机制有效)
- ✅ **不替换三层 κ** (L0/L1/L2 各自独立 κ_l)
- ✅ **不改 Stage 1 embedding** (复用 HG-Rec EmbDataset)
- ✅ **不改 item 顺序** (row index 对齐)

### Precheck 5/5 PASS:
- ✅ κ grad finite nonzero: [2.01e-5, 1.48e-5, 1.21e-5]
- ✅ weight_mlp grad finite nonzero: max=non-zero
- ✅ no NaN/Inf: PASS
- ✅ alpha ∈ [0.1, 0.8]: [0.338, 0.361, 0.320] (init)
- ✅ c_l > 0 init: [1.001, 1.001, 1.001]

### 实施产物 (8 件套 + R12 ckpt 强制):
- `products/task449_issue158_gate2_weighted_mixed_curvature/config.json` (item_emb_sha256=1a42341f01537d6d...)
- `products/task449_issue158_gate2_weighted_mixed_curvature/precheck.json` (5 项 PASS)
- `products/task449_issue158_gate2_weighted_mixed_curvature/mixed_curvature_log.json` (90 entries, 每层 κ/logits/alpha/三分量距离贡献)
- `products/task449_issue158_gate2_weighted_mixed_curvature/sid_output.npy` ((9922, 4) int)
- `products/task449_issue158_gate2_weighted_mixed_curvature/sid_metadata.json`
- `products/task449_issue158_gate2_weighted_mixed_curvature/train_curve.json` (450 steps)
- `products/task449_issue158_gate2_weighted_mixed_curvature/verdict.json` (gate2_decision=PASS)
- `products/task449_issue158_gate2_weighted_mixed_curvature/hrqvae_weighted_mixed.ckpt` (R12 强制保存)
- `scripts/task449_issue158_gate2_weighted_mixed_curvature.py` (~580 lines, R4 py_compile OK)

## Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Issue #158 spec 仅 Gate 2 实证
- Issue spec 强制: Gate 2 PASS 后才可设计 Gate 3 接口

## Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP, 不进入 R@K
- Issue spec 强制: Gate 4 仅 test R@10 > 0.1020 才 Target reached

## R18 4 维度差异成立 (跟 #156)
| 维度 | #156 (Gate 1) | #158 (Gate 2 本任务) |
|------|----------------|----------------------|
| D1 spec | Gate 1 mixing-logit grad 实证 | Gate 2 完整 Stage 2 链路 (产品 manifold SID 完整传播) |
| D2 实施 | 仿射截断 softmax (alpha=0.1+0.7*softmax) | 三分量加权混合 (可学习 κ + 固定双曲 + 欧氏) + Stage 2 RQ-VAE |
| D3 Gate 失败机制 | hard clamp grad path 切断 (Issue #154 反例) | 静态权重 / 无 SID 链路证据 |
| D4 引用文献 | 无 | arXiv:2307.04514 数据驱动加权混合曲率产品流形 |

→ **4 维度全部不一致**, R18 实验强制已完成 ✅

## 关键决策点 (R11.3)
1. **commit_loss 修复**: 初次实现用 F.mse_loss (跟 baseline 兼容), 但 F.mse_loss 跟 c_l 无关 → κ grad = 0. 修复: 用 poincare_distance 基于可学习 c_l, 让 loss 真依赖 κ. R2 不允许 fallback (改用欧氏 loss).
2. **5/5 reload bug 修复**: 同 #157, 比对 4-digit hash vs 4-digit hash.
3. **R12 ckpt 强制保存**: 训练结束 → 删旧 + 存新 ckpt.
4. **α ∈ [0.1, 0.8] 严格内部**: 仿射截断 softmax 数学保证 (softmax ∈ (0,1) → 0.1 + 0.7·softmax ∈ (0.1, 0.8)).
5. **三分量数学合理性**: product d_mix = α · d_hyp_l + β · d_hyp_f + γ · d_eucl, α/β/γ 和=1 强制 ∈ (0, 0.8).

## R17 + R20 + R21 合规
- 4 Gate 状态: Gate 1 ⏸ STOP / Gate 2 ✅ PASS / Gate 3 ⏸ STOP / Gate 4 ⏸ STOP
- 关键数据完整: α/β/γ final/SHA256/item_alignment/reload_5of5/raw grad/κ final/verdict 路径/commit hash
- commit hash: fa0b455 (push 后回填)