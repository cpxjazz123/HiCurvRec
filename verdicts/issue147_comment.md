## Issue #147 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 RQ-VAE / HRQVAE): N/A per spec
- **状态**: Issue #147 spec 仅要求 Gate 3 (Stage 3 T5-mini) 验证, 复用 Task #84 RQ-VAE SID 作为输入
- **Issue spec 强制**: Issue #147 是方向 C Gate 3 验证, 不涉及 Stage 1 RQ-VAE 训练

### Gate 2 (= Stage 2 Sinkhorn + dedup): N/A per spec
- **状态**: Issue #147 spec 不要求 Gate 2
- **Issue spec 强制**: 复用 Task #84 SID 已经过 Sinkhorn 处理 (4-digit unique)

### Gate 3 (= Stage 3 T5-mini): ❌ FAIL (Precheck 3/3 PASS + Training FAIL)
- **关键数据**:
  - **Precheck 1 PASS** ✅: gate=0 → forward 跟原T5 max diff=0 (is_zero_diff=True, 严格 identity-preserving)
  - **Precheck 2 PASS** ✅: 13 个 adapter 参数 grad 有限非零 (gate_logit=7.0083e-03, scale_head.2.bias=6.8877e-03, conditioner/sid_embed/kappa_embed 全部非零)
  - **Precheck 3 PASS** ✅: 真实 history-SID token range 4 层全部 in [0, K_l) (l0/l1/l2/l3 all_in_range=True)
  - **Stage 3 10 epoch training 实际跑完** (41,200 batches total):
    - epoch 0: avg_loss=1.8863, avg_adapter_grad_norm=1.9185e-04
    - epoch 1: avg_loss=1.8886, avg_adapter_grad_norm=2.9376e-08 (**-99.98% 立即坍缩**)
    - epoch 2-9: avg_loss=1.886-1.889 (flat), grad_norm=1e-9 至 1e-8 (持续近零)
    - T5 main loss 完全未下降 (1.886 → 1.888 ≈ 噪声级别)
  - **save/load**: missing=0, unexpected=0 (round-trip 完美)
  - **forward 一致性**: forward_diff=0 (deterministic 验证通过)
- **失败原因**:
  1. **核心根因 — sigmoid 饱和 + frozen T5 切断 adapter 梯度**: scale_head.sigmoid 输出 ∈ [0, 1], 乘以 x_emb 后 residual 极小 → 通过 sigmoid 反向梯度乘以 sigmoid'(x)=sigmoid(x)·(1-sigmoid(x)), sigmoid 接近 0 时 gradient 极小
  2. **次级根因 — frozen T5 main 损失路径**: T5 weights 完全冻结 (只训练 adapter), 损失只通过 residual 路径反向, residual 实际 = 0 后 grad 路径断
  3. **precheck vs training gap**: precheck 一次性 backward 验证 graph 非零, 但 SGD step 后 scale_head.2.bias 朝 0 移动 → sigmoid 输出 → 0 → 所有路径梯度消失
  4. **新 mechanism (跟 #145/#146 不同)**: #145 minimize collapse, #146 argmin+relu 切断, **#147 sigmoid 饱和 + frozen main 反作用**
- **verdict 路径**: `verdicts/task437_issue147_gate3_fail_v1.md`
- **commit**: `1a3c623`

### Gate 4 (= Stage 4 R@K eval): ⏸ STOP per spec
- **原因**: Gate 3 FAIL per spec (Issue #147 spec 明确: Gate 3 PASS 前不做 Stage 4)
- **Issue spec 强制**: Issue #147 spec 仅要求 Gate 3 验证 (Precheck + 短训练), 不要求 Stage 4

### 关键产物
- **commit hash**: `1a3c623`
- **R21 fix**: `08425f2` (R21 强制填入 commit hash)
- **push**: origin/main (R15 强制)
- **verdict**: `verdicts/task437_issue147_gate3_fail_v1.md`
- **实施**: `scripts/task437_issue147_curvature_conditioned_residual.py`
- **8 件套审计**: config.json + SHA256(SID=`2dab2922` + T5=`56d046db` + train=`2c5f843d` + adapter=`6a377018`) + adapter_init_proof + gradient_proof + train_trace (10 epoch 完整) + adapter.pt (save/load round-trip 完美 + forward_diff=0)
- **整体决策**: ⚠️ **架构级 PASS + 训练级 FAIL** — Issue #147 关闭, 不重启 curvature-conditioned residual 路径

### R18 4 维度对比 vs #142 (Issue #129 旧 dual-gate)
- D1 spec: curvature-conditioned residual vs zero-init dual-gate (完全不同)
- D2 实施: MLP + sigmoid scale vs AdapterHookedHGRec.zero-init (完全不同)
- D3 失败机制: sigmoid 饱和 + frozen main vs zero-init 早期 dead (不同 mechanism, 同 family)
- D4 文献: arXiv:2309.04082《Curve Your Attention》 vs #129 内部 spec (完全不同)

### 累计 NO-GO 收口
- 12 方向 NO-GO 收口 + Issue #145/#146/#147 架构级 PARTIAL = **baseline recipe 路径 + 修复尝试 联合耗尽**
- **新方向候选** (R11.5 决策, 等 owner 拍板): train adapter+T5 jointly / linear scale (NOT sigmoid) / regularizer-style aux / Stage 4 protocol split / Stage 1 几何层 conditioning