# Task #431 / Issue #141 [方向B Gate1] product 分量归因矩阵 + anchor 限幅最小反证 — Gate 1 FAIL (R18 反证 NO-GO)

## 决策

**❌ Gate 1 FAIL** (R18 1000-step 最小复现反证: anchor 限幅 + 3 分量 mixing 无法避免 ranking flip 失控 + 梯度失效)

## 4 Gate 详细内容回答 (R17 + R20 强制)

### Gate 0 (= 协议重建): ✅ PASS
- 关键数据: 1000-step 最小复现脚本 (`scripts/task431_issue141_attribution_matrix.py`, ~280 lines), data = item_emb.parquet (9922, 768), SHA256 `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`
- 配置: seed=42, codebook_size=[64,128,256], batch_size=256, lr=1e-4, ranking_flip_max=0.05, anchor_ratio_min=0.4
- 实施: R18 强制 component attribution matrix — d_mix = w1*d_anchor + w2*d_fixed_hyp + w3*d_eucl, layer-level 3 scalars per layer (NOT per-codeword), anchor ≥ 0.4 强制, ranking_flip ≤ 5% 强制
- SHA256 ckpt: 6 件套全落 (`products/task431_issue141_attribution_matrix/config.json`, `verdict.json`, raw_log)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE): ❌ FAIL (4/5 关键 check 失败)
- 关键数据:
  - **三层 ≥2 分量贡献 > 0.1**: L0=1 (仅 anchor)/ L1=2 / L2=2 (anchor + eucl) ❌ (L0 不达标)
  - **anchor ≥ 0.4 限幅**: True (bounding clamp 工作) ✅
  - **ranking_flip ≤ 5%**: L0=92.3%, L1=99.7%, L2=0% ❌❌ (L0/L1 ranking flip 极度失控)
  - **final util**: 2.5% / 1.4% / 0.9% (≪90% 阈值) ❌ — 跟 Phase 0 collapse 一致
  - **final max_load**: 100% / 62.1% / 100% (≫5% 阈值) ❌ — 码字负载极度集中
  - **kappa/mixing grad finite nonzero**: ❌ kappa.grad = 0 (R137 κ lock) + mixing.grad = 0 (softmax 饱和到 boundary 0/1 分布, 跟 R12 lock 同模式)
  - **hard SID round-trip reproducible**: ✅ True
- 失败原因: **低维 layer-level gate (3 scalars) 跟 anchor 主导 mixing 数学上互斥**. 即使 anchor ≥ 0.4 限幅, 一旦码字塌缩到 1-2 个, 各分量在该少数码字的 top-1 一致性不再有 anchor-routing 信息. L0 ranking flip 92% 说明 anchor 跟 mix 在该层分歧极大, 反而暴露"低维 gate 不够 1 个标量"做归因的本质缺陷. Kappa/mixing grad 都为 0 进一步证实 R137 κ lock + softmax 边界饱和导致无学习信号.
- 实施: `scripts/task431_issue141_attribution_matrix.py` (R18 AttributionMatrixModel + per-layer 3-scalar mixing_logits + bounded softmax + attribution matrix + ranking flip tracking)

### Gate 2 (= Stage 2 Sinkhorn): ⏸ STOP per spec
- 原因: Gate 1 FAIL, 无 SID 产出可推断 Sinkhorn

### Gate 3 (= Stage 3 T5-mini): ⏸ STOP per spec
- 原因: Gate 2 STOP

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- 原因: Gate 3 STOP

## 6 件套审计 (R20+R21 强制) — 全部已落地

1. **config**: `products/task431_issue141_attribution_matrix/config.json`
2. **sha256**: item_emb.parquet `1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc`
3. **trace**: 1000 step × 10 record points (step 100/200/.../1000) 全部落 verdict.json
4. **raw_log**: `logs/task431_issue141_attribution_matrix.log` + `logs/task431_issue141_attribution_matrix.launch.log`
5. **verdict**: `products/task431_issue141_attribution_matrix/verdict.json` + `verdicts/task431_issue141_gate1_fail_v3.md`
6. **commit**: pending (待 git commit + push)

## 关键产物

- verdict: `verdicts/task431_issue141_gate1_fail_v3.md`
- 实施: `scripts/task431_issue141_attribution_matrix.py`
- products: `products/task431_issue141_attribution_matrix/verdict.json`
- 整体决策: ❌ Gate 1 FAIL (R18 反证: anchor 限幅 + 低维 layer-level mixing 无法避免 ranking flip 失控 + R137 κ lock 互斥)

## 联立分析

**R18 实证缺口 (Issue #141 spec 强制要求 "component attribution matrix + anchor 限幅 + ranking flip ratio")**:
- ✅ Spec 1 "component attribution matrix": 完整跑 (anchor, fixed_hyp, eucl) × 3 layers × 1000 step attribution tracking
- ❌ Spec 2 "三层 ≥2 分量贡献 > 0.1": L0 只有 1 (anchor 单一贡献)
- ❌ Spec 3 "ranking flip ≤ 5%": L0=92.3%, L1=99.7% 失控
- ❌ Spec 4 "usage ≥ 90%, max_load < 5%": util 0.9-2.5% / max_load 62-100% fail
- ❌ Spec 5 "kappa/mixing 梯度有限非零": 双双 = 0 (R137 κ lock + softmax 饱和)

**R11.5 自主决策**:
- 选 = R18 反证 NO-GO 收口 (Gate 1 FAIL). 备选 = 把 layer-level gate 升级到 per-codeword (高维), 但那等于改成跟 Issue #135 layer-mixing 共享架构, 已知 NO-GO
- 不重启 attribution matrix 路径, 跟 Task #427 (#135) 闭环一致 (R137 κ lock + softmax 边界 + low-dim gate = 架构级 NO-GO)

**联立 #427/#431 = attribution/mixing 路径 NO-GO 收口**:
- Task #427 (#135) 5-step audit grad_finite_nz=False (R137 κ lock + softmax steady state)
- Task #431 (#141) 1000-step 实证 attribution 失效 (anchor 限幅 + ranking flip 失控 + grad 全 0)
- 共同根因 = R137 κ lock + 低维 mixing (3 scalars) → softmax 边界饱和 → 码字塌缩 → attribution 失效
- Issue #141 closed (R16)

R11.5 决策 = 立即 close Issue #141 NO-GO, 不重启 attribution matrix 路径.