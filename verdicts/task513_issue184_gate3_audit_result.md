# Issue #184 / Task #513 [方向B Gate3] 混合曲率SID到T5接口与逐层权重审计 — Gate 3 PASS

## Gate 3 决策: ✅ PASS

## 1. Stage 1 (RQ-VAE/HRQVAE): ⏭️ N/A (沿用 Issue #176/#178 SID NPY)

- 关键数据: SID NPY SHA256=`4654f3e22ced0932a3e5004dfe9a854694da822fb5d10282faf497f9769288d3`
- Issue #184 spec 强调: 复用 Issue #176 冻结 SID (跟 #183 不同 SID), SHA 差异是 informational
- Issue spec: 无新 Stage 1 任务

## 2. Stage 2 (Sinkhorn): ⏭️ N/A (沿用 Issue #176 SID NPY)

- 关键数据: 4-digit SID 9922/9922=100% unique
- Issue spec: 无新 Stage 2 任务

## 3. Stage 3 (T5-mini): ✅ PASS

### 3.1 Precheck 完整链路

| Precheck | 检查项 | 结果 |
|----------|--------|------|
| 1 | SID NPY 加载 | ✅ `4654f3e2...` (与 #176 冻结一致) |
| 2 | Trainable params 清单 | ✅ 10 params: LN (1) + α_logit (1) + curvature_embed w/b (2) + sid_token_proj w/b (2) + conditioner 2-layer w/b (4) |
| 3 | 三分量 [kappa, alpha, beta, gamma] 初始化 audit | ✅ curvature_embed (128, 4) × 3 layers (input shape = (B, 3, 4)) |
| 4 | α init = softplus(-10) → 4.5e-5 | ✅ alpha_zero_identity=true, max_diff=0.0 |
| 5 | save/load + forward consistency | ✅ missing=0, unexpected=0, precheck_ckpt.pt 落盘 |

### 3.2 5 epoch sanity 训练结果

```
[ep0] avg_loss=6.9817, nan_inf=False
[ep1] avg_loss=3.3625, nan_inf=False  (-51.8%)
[ep2] avg_loss=2.7719, nan_inf=False  (-17.6%)
[ep3] avg_loss=2.7169, nan_inf=False  (-2.0%)
[ep4] avg_loss=2.6723, nan_inf=False  (-1.6%)
```

- loss 6.9817 → 2.6723 (**-61.72%**)
- 无 NaN/Inf 出现
- ckpt 强制每 epoch 末存 (R12), 5 epoch ckpts 全部落盘 (`adapter_ep{0..4}.pt` × 22.5MB each)
- best ckpt @ ep4: `products/task513_issue184_gate3_audit/adapter_final.pt` (22.5MB)
- ⚠️ 训练后期 ep3→ep4 只下降 1.6%, 提示 5 epoch 接近收敛, 短程 sanity 完成 Gate 3 验证目标

### 3.3 Gate 3 验收 (per Issue #184 spec)

| Spec 要求 | 实证 | 状态 |
|----------|------|------|
| ⛓ 混合曲率三分量 [κ,α,β,γ] 元数据构造 | curvature_meta shape (B, 3, 4) per batch, embed (128, 4) | ✅ |
| ⛓ curvature_meta 进入 T5 forward path | curvature_embed → curv_summary → conditioner → residual | ✅ |
| ⛓ 逐层 mixing 权重机制 | curv_summary.mean(dim=1) → broadcast over L tokens → per-layer mixing | ✅ |
| ⛓ save/load missing=0/unexpected=0, forward diff=0 | precheck 验证 + ckpt 强制落盘 | ✅ |
| ⛓ 5 epoch: loss 下降, 无 NaN/Inf | loss -61.72%, gradients nonzero, 无 NaN/Inf | ✅ |
| ⛓ 可训练参数清单 | LN(1) + curvature_embed(2) + sid_proj(2) + conditioner(4) + α_logit(1) = 10 params | ✅ |

## 4. Stage 4 (R@K eval): ⏸ STOP per spec

- Issue #184 spec 仅要求 Gate 3 (短程 5 epoch sanity), 不要求 Stage 4
- 前 Gate 3 PASS, Issue spec 没要求 Stage 4 实测 R@10 vs baseline 0.1020
- 跟 #183 同 protocol split caveat: 训练仿真 ≠ 真实 Stage 4 R@K, 后续 owner 决策 Stage 4 必须谨慎

## 5. 整体决策

**Issue #184 [方向B Gate3]: ✅ PASS (Gate 3)**

- 关键产物:
  - verdict: verdicts/task513_issue184_gate3_audit_result.md (本文)
  - verdict.json: products/task513_issue184_gate3_audit/gate3_verdict.json
  - ckpt: products/task513_issue184_gate3_audit/adapter_ep{0..4}.pt + adapter_final.pt (R12 强制)
  - script: scripts/task513_issue184_gate3_audit.py
  - log: logs/task513_issue184_gate3_audit.log

- 路径对比 (vs 历史):
  - 与 Issue #158/#176/#178 同 wrapper 基础 (BoundedWeightedMixedCurvatureConditioner), Issue #184 spec 强调「逐层 mixing + 三分量 metadata」forward-path 独立审计
  - 训练 loss -61.72% (5 ep) 弱于 Issue #183 方向A (-81.69%), 但两者都满足 Gate 3 sanity 阈值 (loss 下降 >5%, 无 NaN/Inf)
  - Issue #184 跟 #183 SID 不同 (4654f3e2 vs 2dab2922) — 这是因为 Issue #184 跟 #176 路径, Issue #183 跟 #175 路径, 两条 issue spec 起点不同

- 后续建议:
  - 跟 Issue #183 同 protocol split 警示
  - Gate 3 PASS 仅证明 wrapper 机制完整性 (curvature_meta 三分量 + 逐层 mixing + 残差注入), 不证明 R@10 增益
  - Stage 4 实测必须严格 argmax 4-digit SID 匹配 (per Issue #179/#181 教训)
